#ifndef EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_PARALLELIZE
#endif

#include "lta/lta.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace lta {
namespace {

static_assert(EIGEN_WORLD_VERSION == 3 && EIGEN_MAJOR_VERSION == 4 &&
                  EIGEN_MINOR_VERSION == 0,
              "This core is pinned to Eigen 3.4.0");

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kPi = 3.14159265358979323846;

struct PCG {
    std::uint64_t state;
    explicit PCG(std::uint64_t s) : state(s ? s : 0x9e3779b97f4a7c15ULL) {}
    std::uint32_t next() {
        std::uint64_t old = state;
        state = old * 6364136223846793005ULL + 1442695040888963407ULL;
        std::uint32_t xs = static_cast<std::uint32_t>(((old >> 18u) ^ old) >> 27u);
        std::uint32_t rot = static_cast<std::uint32_t>(old >> 59u);
        return (xs >> rot) | (xs << ((-static_cast<int>(rot)) & 31));
    }
    double uni() { return (static_cast<double>(next()) + 0.5) / 4294967296.0; }
};

std::uint64_t mix_seed(std::uint64_t master, std::uint64_t id) {
    std::uint64_t z = master + 0x9E3779B97F4A7C15ULL * (id + 1ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double logsumexp(const double* x, int n) {
    double m = -kInf;
    for (int i = 0; i < n; ++i) m = std::max(m, x[i]);
    if (!std::isfinite(m)) return -kInf;
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::exp(x[i] - m);
    return m + std::log(s);
}

void dirichlet(PCG& rng, double* out, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        out[i] = -std::log(std::max(rng.uni(), 1e-16));
        s += out[i];
    }
    for (int i = 0; i < n; ++i) out[i] /= s;
}

int n_tau_blocks(const Panel& d, const Spec& s) {
    if (d.T < 2) return 0;
    return s.transition_equal ? 1 : (d.T - 1);
}
int n_meas_blocks(const Panel& d, const Spec& s) {
    return s.measurement_invariant ? 1 : d.T;
}
int n_mean_blocks(const Panel& d, const Spec& s) {
    return s.means_invariant ? 1 : d.T;
}
int n_cov_blocks(const Panel& d, const Spec& s) {
    return s.covariance_invariant ? 1 : d.T;
}
int packed_size(int J) {
    if (J < 0) throw std::invalid_argument("packed covariance has a negative dimension");
    const std::size_t n = static_cast<std::size_t>(J) * static_cast<std::size_t>(J + 1) / 2u;
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("packed covariance exceeds the supported size");
    }
    return static_cast<int>(n);
}
int cov_block_size(int model, int K, int J) {
    if (model == COV_EEI) return J;
    if (model == COV_EEE) return packed_size(J);
    const std::size_t one = static_cast<std::size_t>(model == COV_VVI ? J : packed_size(J));
    if (K < 0) throw std::invalid_argument("covariance block has a negative dimension");
    if (one != 0 && static_cast<std::size_t>(K) > std::numeric_limits<std::size_t>::max() / one) {
        throw std::invalid_argument("covariance block exceeds the supported size");
    }
    const std::size_t n = static_cast<std::size_t>(K) * one;
    if (model != COV_VVI && model != COV_VVV) throw std::invalid_argument("unsupported covariance model");
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("covariance block exceeds the supported size");
    }
    return static_cast<int>(n);
}

std::size_t cell(int i, int t, int j, int T, int J) { return cell_index(i, t, j, T, J); }

void require_panel(const Panel& d) { (void)indicator_layout(d); }

bool missing_at(const Panel& d, const IndicatorLayout& L, int i, int t, int j) {
    if (L.family[j] == FAM_GAUSSIAN) {
        return !std::isfinite(d.gauss[cell(i, t, j, d.T, d.J)]);
    }
    return d.cat[cell(i, t, j, d.T, d.J)] < 0;
}

bool subject_has_data(const Panel& d, const IndicatorLayout& L, int i) {
    for (int t = 0; t < d.T; ++t) {
        for (int j = 0; j < d.J; ++j) {
            if (!missing_at(d, L, i, t, j)) return true;
        }
    }
    return false;
}

std::string hash_panel(const Panel& d) {
    std::uint64_t h = 14695981039346656037ULL;
    auto mixb = [&](unsigned char b) {
        h ^= b;
        h *= 1099511628211ULL;
    };
    auto mix_int = [&](int v) {
        for (int k = 0; k < 4; ++k) mixb(static_cast<unsigned char>((v >> (8 * k)) & 255));
    };
    const IndicatorLayout L = indicator_layout(d);
    mix_int(d.N);
    mix_int(d.T);
    mix_int(d.J);
    mix_int(L.measurement);
    for (int f : L.family) mix_int(f);
    for (int nc : d.ncat) mix_int(nc);
    if (L.n_categorical) {
        for (int v : d.cat) mix_int(v);
    }
    if (L.n_gaussian) {
        for (double v : d.gauss) {
            if (!std::isfinite(v)) {
                mixb(0);
            } else {
                const auto* p = reinterpret_cast<const unsigned char*>(&v);
                for (int k = 0; k < static_cast<int>(sizeof(double)); ++k) mixb(p[k]);
            }
        }
    }
    std::ostringstream os;
    os << std::hex << h;
    return os.str();
}

void load_sigma(const Params& p, int model, int K, int J, int block, int k,
                Eigen::MatrixXd& S) {
    S.setZero(J, J);
    const int bsz = cov_block_size(model, K, J);
    const double* base = p.cov.data() + static_cast<std::size_t>(block) * static_cast<std::size_t>(bsz);
    if (model == COV_EEI) {
        for (int j = 0; j < J; ++j) S(j, j) = base[j];
    } else if (model == COV_VVI) {
        const double* v = base + static_cast<std::size_t>(k) * static_cast<std::size_t>(J);
        for (int j = 0; j < J; ++j) S(j, j) = v[j];
    } else if (model == COV_EEE) {
        for (int a = 0; a < J; ++a)
            for (int b = 0; b <= a; ++b) {
                double v = base[packed_index(a, b)];
                S(a, b) = S(b, a) = v;
            }
    } else {
        const double* pk = base + static_cast<std::size_t>(k) * static_cast<std::size_t>(packed_size(J));
        for (int a = 0; a < J; ++a)
            for (int b = 0; b <= a; ++b) {
                double v = pk[packed_index(a, b)];
                S(a, b) = S(b, a) = v;
            }
    }
}

void store_sigma_block(std::vector<double>& cov, int model, int K, int J, int block,
                       int k_or_neg1, const Eigen::MatrixXd& S) {
    const int bsz = cov_block_size(model, K, J);
    double* base = cov.data() + static_cast<std::size_t>(block) * static_cast<std::size_t>(bsz);
    if (model == COV_EEI || model == COV_VVI) {
        double* v = (model == COV_EEI) ? base
                                       : base + static_cast<std::size_t>(k_or_neg1) * static_cast<std::size_t>(J);
        for (int j = 0; j < J; ++j) v[j] = S(j, j);
    } else {
        double* pk = (model == COV_EEE)
                         ? base
                         : base + static_cast<std::size_t>(k_or_neg1) * static_cast<std::size_t>(packed_size(J));
        for (int a = 0; a < J; ++a)
            for (int b = 0; b <= a; ++b) pk[packed_index(a, b)] = S(a, b);
    }
}

double min_eig(const Eigen::MatrixXd& S) {
    if (S.size() == 0) return kNaN;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S, Eigen::EigenvaluesOnly);
    if (es.info() != Eigen::Success) return -kInf;
    return es.eigenvalues().minCoeff();
}

double max_eig(const Eigen::MatrixXd& S) {
    if (S.size() == 0) return kNaN;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S, Eigen::EigenvaluesOnly);
    if (es.info() != Eigen::Success) return kInf;
    return es.eigenvalues().maxCoeff();
}

// Correlation-matrix eigenvalue below this is a collapsed component.
// The EM floor uses the same dimensionless cutoff. It does not look at the
// absolute eigenvalue, so a change of units does not trip it.
constexpr double kCorrFloor = 1e-8;
constexpr double kCondWarn = 1e6;
constexpr double kRelVarFloor = 1e-8;
constexpr double kNumericalEmpty = 1e-8;

bool correlation_needs_floor(const Eigen::MatrixXd& S, double* min_corr) {
    const int p = static_cast<int>(S.rows());
    if (p == 0) return false;
    for (int j = 0; j < p; ++j) {
        if (!(S(j, j) > 0.0) || !std::isfinite(S(j, j))) {
            if (min_corr) *min_corr = -kInf;
            return true;
        }
    }
    Eigen::MatrixXd R(p, p);
    for (int a = 0; a < p; ++a) {
        double sa = std::sqrt(S(a, a));
        for (int b = 0; b < p; ++b) {
            double sb = std::sqrt(S(b, b));
            R(a, b) = S(a, b) / (sa * sb);
        }
    }
    R = 0.5 * (R + R.transpose());
    double m = min_eig(R);
    if (min_corr) *min_corr = m;
    return !(m > kCorrFloor);
}

bool ridge_if_needed(Eigen::MatrixXd& S, double* global_min) {
    double m = min_eig(S);
    if (global_min) *global_min = std::min(*global_min, m);
    if (!correlation_needs_floor(S, nullptr)) return false;
    const int p = static_cast<int>(S.rows());
    double log_scale = 0.0;
    int npos = 0;
    for (int j = 0; j < p; ++j) {
        if (S(j, j) > 0.0 && std::isfinite(S(j, j))) {
            log_scale += std::log(S(j, j));
            ++npos;
        }
    }
    // 1e-8 of the matrix's own geometric-mean variance. A well-conditioned
    // matrix whose entries are small only because of units is not floored.
    double scale = npos ? std::exp(log_scale / static_cast<double>(npos)) : 1.0;
    double floor_var = kCorrFloor * scale;
    for (int j = 0; j < p; ++j) {
        if (!(S(j, j) > floor_var)) S(j, j) = floor_var;
    }
    if (correlation_needs_floor(S, nullptr)) S.diagonal().array() += floor_var;
    return true;
}

Eigen::VectorXd solve_spd(const Eigen::MatrixXd& L, const Eigen::VectorXd& b) {
    Eigen::VectorXd z = L.triangularView<Eigen::Lower>().solve(b);
    return L.transpose().triangularView<Eigen::Upper>().solve(z);
}

// Lower Cholesky of Sigma, formed from the correlation matrix so a change of
// units does not make a well-conditioned matrix look singular. A non-positive
// variance or a collapsed correlation eigenvalue is floored and flagged.
Eigen::MatrixXd factor_cov(const Eigen::MatrixXd& Sigma, bool* regularized) {
    Eigen::MatrixXd S = 0.5 * (Sigma + Sigma.transpose());
    const int p = static_cast<int>(S.rows());
    bool pos = p > 0;
    Eigen::VectorXd scale(p);
    for (int a = 0; a < p; ++a) {
        if (!(S(a, a) > 0.0) || !std::isfinite(S(a, a))) pos = false;
        else scale(a) = std::sqrt(S(a, a));
    }
    if (pos) {
        Eigen::MatrixXd R(p, p);
        for (int a = 0; a < p; ++a)
            for (int b = 0; b < p; ++b) R(a, b) = S(a, b) / (scale(a) * scale(b));
        R = 0.5 * (R + R.transpose());
        Eigen::LLT<Eigen::MatrixXd> chol(R);
        if (chol.info() == Eigen::Success) {
            Eigen::MatrixXd L = chol.matrixL();
            return Eigen::MatrixXd(scale.asDiagonal() * L);
        }
    }
    if (regularized) *regularized = true;
    Eigen::MatrixXd ridged = S;
    ridge_if_needed(ridged, nullptr);
    Eigen::LLT<Eigen::MatrixXd> chol(ridged);
    if (chol.info() != Eigen::Success) {
        double log_s = 0.0;
        int n = 0;
        for (int a = 0; a < p; ++a)
            if (ridged(a, a) > 0.0) {
                log_s += std::log(ridged(a, a));
                ++n;
            }
        double add = n ? 1e-6 * std::exp(log_s / static_cast<double>(n)) : 1e-6;
        ridged.diagonal().array() += add;
        chol.compute(ridged);
    }
    return chol.matrixL();
}

struct Key {
    int t;
    int k;
    std::uint64_t mask;
    bool operator==(const Key& o) const { return t == o.t && k == o.k && mask == o.mask; }
};
struct KeyHash {
    std::size_t operator()(const Key& p) const {
        return std::hash<std::uint64_t>{}(p.mask) ^ (std::hash<int>{}(p.t) << 1) ^
               (std::hash<int>{}(p.k) << 2);
    }
};

// Cholesky factor of one observed submatrix, plus the conditional-moment
// terms that depend only on that factor. The log determinant is stored with
// the factor so an emission does not recompute it. Conditional terms are
// filled on the first incomplete pattern. The cache is cleared after the
// M-step, because the next E-step sees new parameters.
struct GaussFactor {
    Eigen::MatrixXd L;
    double logdet = 0;
    bool regularized = false;
    std::vector<int> obs;
    bool conditional_ready = false;
    Eigen::MatrixXd Smo;
    Eigen::MatrixXd Cmm;
};

struct FactorCache {
    std::unordered_map<Key, GaussFactor, KeyHash> map;
    std::uint64_t param_version = 0;
    void clear() { map.clear(); }
    // A new parameter version drops factors built for the previous parameters.
    void bind(std::uint64_t version) {
        if (version != param_version) {
            map.clear();
            param_version = version;
        }
    }
};

GaussFactor& factor_for(FactorCache& cache, int t, int k, std::uint64_t mask,
                        const Eigen::MatrixXd& Sigma, const std::vector<int>& obs,
                        bool* regularized) {
    Key key{t, k, mask};
    auto it = cache.map.find(key);
    if (it != cache.map.end()) {
        if (it->second.regularized && regularized) *regularized = true;
        return it->second;
    }
    GaussFactor g;
    g.obs = obs;
    const int p = static_cast<int>(obs.size());
    Eigen::MatrixXd Soo(p, p);
    for (int a = 0; a < p; ++a)
        for (int b = 0; b < p; ++b) Soo(a, b) = Sigma(obs[a], obs[b]);
    bool reg = false;
    g.L = factor_cov(Soo, &reg);
    g.regularized = reg;
    if (reg && regularized) *regularized = true;
    g.logdet = 0.0;
    for (int a = 0; a < p; ++a) g.logdet += std::log(std::max(g.L(a, a), 1e-300));
    g.logdet *= 2.0;
    auto placed = cache.map.emplace(key, std::move(g));
    return placed.first->second;
}

double log_gauss_obs(const double* y, const double* mu, const Eigen::MatrixXd& Sigma, int J,
                     FactorCache& cache, int tkey, int kkey, bool* regularized) {
    std::vector<int> obs;
    obs.reserve(J);
    std::uint64_t mask = 0;
    for (int j = 0; j < J; ++j) {
        if (std::isfinite(y[j])) {
            obs.push_back(j);
            mask |= (1ULL << j);
        }
    }
    if (obs.empty()) return 0.0;
    GaussFactor& g = factor_for(cache, tkey, kkey, mask, Sigma, obs, regularized);
    const int p = static_cast<int>(obs.size());
    Eigen::VectorXd d(p);
    for (int a = 0; a < p; ++a) d(a) = y[obs[a]] - mu[obs[a]];
    Eigen::VectorXd z = g.L.triangularView<Eigen::Lower>().solve(d);
    const double quad = z.squaredNorm();
    return -0.5 * (p * std::log(2.0 * kPi) + g.logdet + quad);
}

void conditional_moments(const double* y, const double* mu, const Eigen::MatrixXd& Sigma, int J,
                         Eigen::VectorXd& ey, Eigen::MatrixXd& Eyy, FactorCache& cache, int tkey,
                         int kkey, bool* regularized) {
    std::vector<int> o, m;
    for (int j = 0; j < J; ++j) (std::isfinite(y[j]) ? o : m).push_back(j);
    ey.resize(J);
    Eyy.setZero(J, J);
    for (int j = 0; j < J; ++j) ey(j) = mu[j];
    if (o.empty()) return;
    std::uint64_t mask = 0;
    for (int j : o) mask |= (1ULL << j);
    GaussFactor& g = factor_for(cache, tkey, kkey, mask, Sigma, o, regularized);
    const int no = static_cast<int>(o.size());
    const int nm = static_cast<int>(m.size());
    Eigen::VectorXd diff(no);
    for (int a = 0; a < no; ++a) {
        diff(a) = y[o[a]] - mu[o[a]];
        ey(o[a]) = y[o[a]];
    }
    Eigen::MatrixXd Cov = Eigen::MatrixXd::Zero(J, J);
    if (nm) {
        if (!g.conditional_ready) {
            Eigen::MatrixXd Smo(nm, no), Smm(nm, nm);
            for (int a = 0; a < nm; ++a) {
                for (int b = 0; b < no; ++b) Smo(a, b) = Sigma(m[a], o[b]);
                for (int b = 0; b < nm; ++b) Smm(a, b) = Sigma(m[a], m[b]);
            }
            Eigen::MatrixXd Som(no, nm);
            for (int a = 0; a < no; ++a)
                for (int b = 0; b < nm; ++b) Som(a, b) = Sigma(o[a], m[b]);
            Eigen::MatrixXd A(no, nm);
            for (int c = 0; c < nm; ++c) A.col(c) = solve_spd(g.L, Som.col(c));
            g.Cmm = Smm - Smo * A;
            g.Smo = std::move(Smo);
            g.conditional_ready = true;
        }
        Eigen::VectorXd solved = solve_spd(g.L, diff);
        Eigen::VectorXd dev = g.Smo * solved;
        for (int a = 0; a < nm; ++a) ey(m[a]) = mu[m[a]] + dev(a);
        for (int a = 0; a < nm; ++a)
            for (int b = 0; b < nm; ++b) Cov(m[a], m[b]) = g.Cmm(a, b);
    }
    // A complete observation has an empty missing block. The factor above
    // still runs, so a ridge on the full covariance is reported. Cov stays 0.
    // Return that conditional covariance. Adding ey*ey' and subtracting it
    // later cancels the scatter of observations at a large location.
    Eyy = Cov;
}

struct FB {
    double loglik = -kInf;
    std::vector<double> gamma;
    std::vector<double> filtered;
    std::vector<double> xi;
    std::vector<double> subject;  // per-subject observed log-likelihood; NaN if excluded
    bool regularized = false;
};

// Scratch reused across EM iterations of one start. The factor cache is
// cleared after each M-step. Vectors keep their capacity.
struct ParameterPrep {
    std::uint64_t version = 0;
    std::vector<double> log_delta;
    std::vector<double> log_tau;
    std::vector<double> log_rho;
    bool shared = false;
    int K = 0;
    std::vector<Eigen::MatrixXd> sigma;
    const Eigen::MatrixXd& covariance(int block, int k) const {
        if (shared) return sigma[static_cast<std::size_t>(block)];
        return sigma[span_index(block, K, k)];
    }
};

struct EmWorkspace {
    FactorCache cache;
    std::vector<double> logb;
    std::vector<double> loga;
    std::vector<double> logbe;
    std::vector<double> emis;
    std::vector<double> alpha;
    std::vector<double> beta;
    std::vector<double> log_delta;
    std::vector<double> log_tau;
    std::vector<double> logxi;
    std::vector<double> ybuf;
    Eigen::VectorXd ey;
    Eigen::MatrixXd Eyy;
    Eigen::MatrixXd sharedS;
    Eigen::MatrixXd stateS;
    void invalidate_factors() { cache.clear(); }
};

void emission_row(const Panel& d, const IndicatorLayout& L, const Spec& s, const Params& par,
                  const ParameterPrep& prep, int i, int t, std::vector<double>& logb, double* ybuf,
                  bool* regularized, FactorCache& cache) {
    cache.bind(prep.version);
    const int K = s.K;
    if (static_cast<int>(logb.size()) != K) logb.assign(static_cast<std::size_t>(K), 0.0);
    else std::fill(logb.begin(), logb.end(), 0.0);
    if (L.n_categorical) {
        int mb = s.measurement_invariant ? 0 : t;
        for (int k = 0; k < K; ++k) {
            double lb = 0.0;
            for (int j : L.categorical_index) {
                int y = d.cat[cell(i, t, j, d.T, d.J)];
                if (y < 0) continue;
                const double lp = prep.log_rho[span_index(span_index(span_index(mb, d.J, j), K, k), d.Rmax, y)];
                if (!std::isfinite(lp)) {
                    lb = -kInf;
                    break;
                }
                lb += lp;
            }
            logb[static_cast<std::size_t>(k)] = lb;
        }
    }
    if (!L.n_gaussian) return;
    int mean_b = s.means_invariant ? 0 : t;
    int cov_b = s.covariance_invariant ? 0 : t;
    int tkey = s.covariance_invariant ? -1 : t;
    const int Jg = L.n_gaussian;
    for (int g = 0; g < Jg; ++g) ybuf[g] = d.gauss[cell(i, t, L.gaussian_index[g], d.T, d.J)];
    const bool shared = prep.shared;
    for (int k = 0; k < K; ++k) {
        const double* mu = par.means.data() + span_index(span_index(mean_b, K, k), Jg, 0);
        const int kkey = shared ? -1 : k;
        logb[static_cast<std::size_t>(k)] +=
            log_gauss_obs(ybuf, mu, prep.covariance(cov_b, k), Jg, cache, tkey, kkey, regularized);
    }
}

ParameterPrep prepare_parameters(const Panel& d, const IndicatorLayout& L, const Spec& s, const Params& par);

void prepare_log_probs(const Params& par, int K, std::vector<double>& log_delta,
                       std::vector<double>& log_tau) {
    log_delta.resize(static_cast<size_t>(K));
    for (int k = 0; k < K; ++k)
        log_delta[static_cast<size_t>(k)] = par.delta[static_cast<size_t>(k)] > 0
                                                 ? std::log(par.delta[static_cast<size_t>(k)])
                                                 : -kInf;
    log_tau.resize(par.tau.size());
    for (size_t i = 0; i < par.tau.size(); ++i)
        log_tau[i] = par.tau[i] > 0 ? std::log(par.tau[i]) : -kInf;
}

ParameterPrep prepare_parameters(const Panel& d, const IndicatorLayout& L, const Spec& s, const Params& par) {
    ParameterPrep prep;
    static std::atomic<std::uint64_t> counter{1};
    prep.version = counter.fetch_add(1, std::memory_order_relaxed);
    prep.K = s.K;
    prepare_log_probs(par, s.K, prep.log_delta, prep.log_tau);
    prep.log_rho.resize(par.rho.size());
    for (std::size_t i = 0; i < par.rho.size(); ++i)
        prep.log_rho[i] = par.rho[i] > 0.0 ? std::log(par.rho[i]) : -kInf;
    prep.shared = s.cov_model == COV_EEI || s.cov_model == COV_EEE;
    if (L.n_gaussian) {
        const int ncb = n_cov_blocks(d, s);
        const int Jg = L.n_gaussian;
        if (prep.shared) {
            prep.sigma.resize(static_cast<std::size_t>(ncb));
            for (int b = 0; b < ncb; ++b)
                load_sigma(par, s.cov_model, s.K, Jg, b, 0, prep.sigma[static_cast<std::size_t>(b)]);
        } else {
            prep.sigma.resize(checked_product("flat", ncb, s.K));
            for (int b = 0; b < ncb; ++b)
                for (int k = 0; k < s.K; ++k)
                    load_sigma(par, s.cov_model, s.K, Jg, b, k,
                               prep.sigma[span_index(b, s.K, k)]);
        }
    }
    return prep;
}

// One forward step. `dest` and `emis` have length K. `prev` is the previous
// time's forward messages, unused at t = 0. `loga` is caller scratch of length K.
void forward_time(const double* prev, const double* emis, const std::vector<double>& log_delta,
                  const std::vector<double>& log_tau, int t, int tb, int K, double* dest,
                  double* loga) {
    if (t == 0) {
        for (int k = 0; k < K; ++k)
            dest[k] = log_delta[static_cast<size_t>(k)] + emis[k];
        return;
    }
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < K; ++j)
            loga[j] = prev[j] + log_tau[span_index(span_index(tb, K, j), K, k)];
        dest[k] = emis[k] + logsumexp(loga, K);
    }
}

// Compensated sum. A long chain's log-likelihood is a sum of one O(1) term
// per time step. Adding those terms into a running total near 1e7 drops
// digits; Neumaier keeps the low-order part. The recursion itself stays in
// double precision.
struct NeumaierSum {
    double s = 0.0;
    double c = 0.0;
    void add(double x) {
        const double t = s + x;
        if (std::abs(s) >= std::abs(x)) c += (s - t) + x;
        else c += (x - t) + s;
        s = t;
    }
    double total() const { return s + c; }
};

// Centered log-space forward. Each step subtracts its log-sum-exp and the
// subtracted values are accumulated with NeumaierSum. The messages that the
// backward pass stores are not these centered messages.
double centered_forward_loglik(int T, int K, const double* emis, const std::vector<double>& log_delta,
                               const std::vector<double>& log_tau, bool transition_equal) {
    std::vector<double> cur(static_cast<std::size_t>(K)), nxt(static_cast<std::size_t>(K));
    std::vector<double> loga(static_cast<std::size_t>(K));
    NeumaierSum acc;
    for (int t = 0; t < T; ++t) {
        const double* prev = t == 0 ? nullptr : cur.data();
        const int tb = transition_equal ? 0 : (t - 1);
        forward_time(prev, emis + static_cast<std::size_t>(t) * static_cast<std::size_t>(K), log_delta, log_tau,
                     t, tb, K, nxt.data(), loga.data());
        const double m = logsumexp(nxt.data(), K);
        if (!std::isfinite(m)) return -kInf;
        for (int k = 0; k < K; ++k) cur[static_cast<std::size_t>(k)] = nxt[static_cast<std::size_t>(k)] - m;
        acc.add(m);
    }
    return acc.total();
}

// Wall time of the extra centered forward inside the last forward_backward
// on this calling thread. Parallel starts each call forward_backward, so a
// process-wide double would be an unsynchronized write. Same ownership as
// g_phase: one record per calling thread, not shared across workers.
thread_local double g_last_second_forward_seconds = 0.0;

FB forward_backward(const Panel& d, const Spec& s, const Params& par,
                    const std::vector<int>& analyzed, EmWorkspace* scratch) {
    EmWorkspace local;
    EmWorkspace& ws = scratch ? *scratch : local;
    g_last_second_forward_seconds = 0.0;
    FB fb;
    const IndicatorLayout L = indicator_layout(d);
    const int N = d.N, T = d.T, K = s.K;
    fb.gamma.assign(checked_product("posterior", N, T, K), kNaN);
    fb.filtered.assign(checked_product("posterior", N, T, K), kNaN);
    fb.xi.assign(checked_product("transition posterior", N, T - 1, K, K), kNaN);
    fb.subject.assign(N, kNaN);
    double total = 0.0;
    const ParameterPrep prep = prepare_parameters(d, L, s, par);
    ws.log_delta = prep.log_delta;
    ws.log_tau = prep.log_tau;
    ws.ybuf.resize(static_cast<std::size_t>(L.n_gaussian));
    ws.logb.resize(static_cast<size_t>(K));
    ws.loga.resize(static_cast<size_t>(K));
    ws.logbe.resize(static_cast<size_t>(K));
    ws.alpha.resize(checked_product("flat", T, K));
    ws.beta.resize(checked_product("flat", T, K));
    ws.emis.resize(checked_product("flat", T, K));
    ws.logxi.resize(checked_product("flat", K, K));
    for (int i : analyzed) {
        // One T by K emission per subject. Forward, backward, and the joint
        // transition posterior all read this matrix.
        for (int t = 0; t < T; ++t) {
            emission_row(d, L, s, par, prep, i, t, ws.logb, ws.ybuf.data(), &fb.regularized, ws.cache);
            for (int k = 0; k < K; ++k) ws.emis[span_index(t, K, k)] = ws.logb[static_cast<size_t>(k)];
        }
        for (int t = 0; t < T; ++t) {
            const double* prev = t == 0 ? nullptr : ws.alpha.data() +checked_product("flat", (t - 1), K);
            int tb = s.transition_equal ? 0 : (t - 1);
            forward_time(prev, ws.emis.data() + checked_product("flat", t, K), ws.log_delta, ws.log_tau, t, tb,
                         K, ws.alpha.data() + checked_product("flat", t, K), ws.loga.data());
        }
        const auto second_started = std::chrono::steady_clock::now();
        double ll = centered_forward_loglik(T, K, ws.emis.data(), ws.log_delta, ws.log_tau, s.transition_equal);
        g_last_second_forward_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - second_started).count();
        fb.subject[i] = ll;
        if (!std::isfinite(ll)) {
            fb.loglik = -kInf;
            return fb;
        }
        total += ll;
        for (int k = 0; k < K; ++k) ws.beta[span_index((T - 1), K, k)] = 0.0;
        for (int t = T - 2; t >= 0; --t) {
            int tb = s.transition_equal ? 0 : t;
            for (int j = 0; j < K; ++j) {
                for (int k = 0; k < K; ++k)
                    ws.logbe[static_cast<size_t>(k)] =
                        ws.log_tau[span_index(span_index(tb, K, j), K, k)] +
                        ws.emis[span_index((t + 1), K, k)] +
                        ws.beta[span_index((t + 1), K, k)];
                ws.beta[span_index(t, K, j)] = logsumexp(ws.logbe.data(), K);
            }
        }
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < K; ++k) ws.loga[static_cast<size_t>(k)] = ws.alpha[span_index(t, K, k)];
            double norm = logsumexp(ws.loga.data(), K);
            if (!std::isfinite(norm)) {
                fb.loglik = -kInf;
                fb.subject[i] = -kInf;
                return fb;
            }
            for (int k = 0; k < K; ++k) {
                fb.filtered[flat_index(i, T, t, K, k)] = std::exp(ws.alpha[span_index(t, K, k)] - norm);
                ws.loga[static_cast<size_t>(k)] =
                    ws.alpha[span_index(t, K, k)] + ws.beta[span_index(t, K, k)] - ll;
            }
            // One normalizing constant for the K states. The shift is the
            // largest finite log weight, applied before the exponential.
            double* grow = fb.gamma.data() +flat_index(i, T, t, K, 0);
            if (!normalize_log_weights(ws.loga.data(), K, grow)) {
                fb.loglik = -kInf;
                fb.subject[i] = -kInf;
                return fb;
            }
        }
        for (int t = 0; t < T - 1; ++t) {
            int tb = s.transition_equal ? 0 : t;
            for (int j = 0; j < K; ++j) {
                for (int k = 0; k < K; ++k) {
                    ws.logxi[span_index(j, K, k)] =
                        ws.alpha[span_index(t, K, j)] +
                        ws.log_tau[span_index(span_index(tb, K, j), K, k)] +
                        ws.emis[span_index((t + 1), K, k)] +
                        ws.beta[span_index((t + 1), K, k)] - ll;
                }
            }
            // One normalizing constant for the whole K by K block, not one
            // per origin state.
            double* xrow = fb.xi.data() + flat_index(i, T - 1, t, K, 0) * static_cast<std::size_t>(K);
            if (!normalize_log_weights(ws.logxi.data(), K * K, xrow)) {
                fb.loglik = -kInf;
                fb.subject[i] = -kInf;
                return fb;
            }}
    }
    fb.loglik = total;
    return fb;
}


// Likelihood only. Two length-K buffers roll forward. No smoothed state
// posterior, filtered posterior, or transition posterior is allocated.
// The EM E-step does not call this. It still needs those sufficient statistics.
struct LikelihoodOnly {
    double loglik = -kInf;
    std::vector<double> subject;
    bool regularized = false;
};

LikelihoodOnly forward_likelihood(const Panel& d, const Spec& s, const Params& par,
                                  const std::vector<int>& analyzed) {
    LikelihoodOnly out;
    const IndicatorLayout L = indicator_layout(d);
    const int N = d.N, T = d.T, K = s.K;
    out.subject.assign(static_cast<size_t>(N), kNaN);
    std::vector<double> log_delta, log_tau, logb(static_cast<size_t>(K));
    std::vector<double> cur(static_cast<size_t>(K)), nxt(static_cast<size_t>(K)), loga(static_cast<size_t>(K));
    const ParameterPrep prep = prepare_parameters(d, L, s, par);
    log_delta = prep.log_delta;
    log_tau = prep.log_tau;
    std::vector<double> ybuf(static_cast<std::size_t>(L.n_gaussian));
    FactorCache cache;
    bool regularized = false;
    bool impossible = false;
    double total = 0.0;
    for (int i : analyzed) {
        // Same centered step as centered_forward_loglik, one emission row at a time.
        // Storage stays three length-K buffers. There is no T by K emission matrix.
        NeumaierSum acc;
        bool bad = false;
        for (int t = 0; t < T; ++t) {
            emission_row(d, L, s, par, prep, i, t, logb, ybuf.data(), &regularized, cache);
            const double* prev = t == 0 ? nullptr : cur.data();
            const int tb = s.transition_equal ? 0 : (t - 1);
            forward_time(prev, logb.data(), log_delta, log_tau, t, tb, K, nxt.data(), loga.data());
            const double m = logsumexp(nxt.data(), K);
            if (!std::isfinite(m)) {
                bad = true;
                break;
            }
            for (int k = 0; k < K; ++k) cur[static_cast<size_t>(k)] = nxt[static_cast<size_t>(k)] - m;
            acc.add(m);
        }
        double ll = bad ? -kInf : acc.total();
        out.subject[static_cast<size_t>(i)] = ll;
        if (!std::isfinite(ll)) {
            out.subject[static_cast<size_t>(i)] = -kInf;
            impossible = true;
            continue;
        }
        total += ll;
    }
    out.loglik = impossible ? -kInf : total;
    out.regularized = regularized;
    return out;
}

bool same_bits(double a, double b) {
    static_assert(sizeof(double) == sizeof(std::uint64_t), "double is 64-bit");
    std::uint64_t xa = 0, xb = 0;
    std::memcpy(&xa, &a, sizeof(double));
    std::memcpy(&xb, &b, sizeof(double));
    return xa == xb;
}

std::vector<int> viterbi_path(const Panel& d, const Spec& s, const Params& par, int i,
                               double* log_joint) {
    const IndicatorLayout L = indicator_layout(d);
    const int T = d.T, K = s.K;
    std::vector<double> delta(checked_product("flat", T, K)), logb;
    std::vector<int> psi(checked_product("flat", T, K), 0);
    const ParameterPrep prep = prepare_parameters(d, L, s, par);
    std::vector<double> ybuf(static_cast<std::size_t>(L.n_gaussian));
    FactorCache cache;
    bool reg = false;
    for (int t = 0; t < T; ++t) {
        emission_row(d, L, s, par, prep, i, t, logb, ybuf.data(), &reg, cache);
        if (t == 0) {
            for (int k = 0; k < K; ++k) {
                double ld = par.delta[k] > 0 ? std::log(par.delta[k]) : -kInf;
                delta[k] = ld + logb[k];
            }
        } else {
            int tb = s.transition_equal ? 0 : (t - 1);
            for (int k = 0; k < K; ++k) {
                double best = -kInf;
                int arg = 0;
                for (int j = 0; j < K; ++j) {
                    double lt = par.tau[span_index(span_index(tb, K, j), K, k)];
                    double v = delta[span_index((t - 1), K, j)] + (lt > 0 ? std::log(lt) : -kInf);
                    if (v > best) {
                        best = v;
                        arg = j;
                    }
                }
                delta[span_index(t, K, k)] = best + logb[k];
                psi[span_index(t, K, k)] = arg;
            }
        }
    }
    std::vector<int> path(T, 0);
    int last = 0;
    double best = -kInf;
    for (int k = 0; k < K; ++k)
        if (delta[span_index((T - 1), K, k)] > best) {
            best = delta[span_index((T - 1), K, k)];
            last = k;
        }
    path[T - 1] = last;
    for (int t = T - 1; t > 0; --t) {
        path[t - 1] = psi[span_index(t, K, path[t])];
    }
    if (log_joint) *log_joint = best;
    return path;
}

// Conditional maximization of a mean shared across waves when each wave has
// its own covariance. The weight is n_t * Sigma_t^{-1}. A covariance that is
// not positive definite returns false and the caller keeps the unweighted mean.
bool precision_weighted_mean(const Panel& d, const Spec& s, const Params& current, int Jg, int k,
                             const std::vector<double>& nwt, const std::vector<Eigen::VectorXd>& sumy,
                             Eigen::VectorXd& mean) {
    const int T = d.T;
    const int K = s.K;
    const bool shared = s.cov_model == COV_EEI || s.cov_model == COV_EEE;
    Eigen::MatrixXd info = Eigen::MatrixXd::Zero(Jg, Jg);
    Eigen::VectorXd score = Eigen::VectorXd::Zero(Jg);
    Eigen::MatrixXd sigma(Jg, Jg);
    bool any = false;
    for (int t = 0; t < T; ++t) {
        const double n = nwt[span_index(t, K, k)];
        if (!(n > 0.0)) continue;
        any = true;
        load_sigma(current, s.cov_model, K, Jg, t, shared ? 0 : k, sigma);
        Eigen::LDLT<Eigen::MatrixXd> ldlt(sigma);
        if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) return false;
        const Eigen::MatrixXd prec = ldlt.solve(Eigen::MatrixXd::Identity(Jg, Jg));
        info += n * prec;
        score += prec * sumy[span_index(t, K, k)];
    }
    if (!any) return false;
    Eigen::LDLT<Eigen::MatrixXd> system(info);
    if (system.info() != Eigen::Success || !system.isPositive()) return false;
    mean = system.solve(score);
    return mean.allFinite();
}

Params mstep(const Panel& d, const Spec& s, const Params& current, const FB& fb,
             const std::vector<int>& analyzed, bool* regularized, double* min_eigen,
             EmWorkspace* scratch) {
    const int K = s.K, T = d.T, J = d.J;
    Params out = current;
    // initial
    std::vector<double> init(K, 0.0);
    for (int i : analyzed)
        for (int k = 0; k < K; ++k) init[k] += fb.gamma[flat_index(i, T, 0, K, k)];
    double isum = std::accumulate(init.begin(), init.end(), 0.0);
    out.delta.resize(K);
    if (isum <= 0) {
        for (int k = 0; k < K; ++k) out.delta[k] = 1.0 / K;
    } else {
        for (int k = 0; k < K; ++k) out.delta[k] = init[k] / isum;
    }
    int ntb = n_tau_blocks(d, s);
    if (ntb) {
        std::vector<double> num(checked_product("flat", ntb, K, K), 0.0), den(checked_product("flat", ntb, K), 0.0);
        for (int i : analyzed) {
            for (int t = 0; t < T - 1; ++t) {
                int b = s.transition_equal ? 0 : t;
                for (int j = 0; j < K; ++j) {
                    for (int k = 0; k < K; ++k) {
                        double x = fb.xi[xi_index(i, T - 1, t, K, j, k)];
                        num[span_index(span_index(b, K, j), K, k)] += x;
                        den[span_index(b, K, j)] += x;
                    }
                }
            }
        }
        out.tau.assign(checked_product("flat", ntb, K, K), 0.0);
        for (int b = 0; b < ntb; ++b) {
            for (int j = 0; j < K; ++j) {
                if (den[span_index(b, K, j)] <= 0) {
                    for (int k = 0; k < K; ++k) out.tau[span_index(span_index(b, K, j), K, k)] = 1.0 / K;
                } else {
                    for (int k = 0; k < K; ++k)
                        out.tau[span_index(span_index(b, K, j), K, k)] = num[span_index(span_index(b, K, j), K, k)] / den[span_index(b, K, j)];
                }
            }
        }
    } else {
        out.tau.clear();
    }
    const IndicatorLayout L = indicator_layout(d);
    if (L.n_categorical) {
        int nmb = n_meas_blocks(d, s);
        std::vector<double> num(checked_product("flat", nmb, J, K, d.Rmax), 0.0),
            den(checked_product("flat", nmb, J, K), 0.0);
        for (int i : analyzed) {
            for (int t = 0; t < T; ++t) {
                int b = s.measurement_invariant ? 0 : t;
                for (int j : L.categorical_index) {
                    int y = d.cat[cell(i, t, j, T, J)];
                    if (y < 0) continue;
                    for (int k = 0; k < K; ++k) {
                        double g = fb.gamma[flat_index(i, T, t, K, k)];
                        num[span_index(span_index(span_index(b, J, j), K, k), d.Rmax, y)] += g;
                        den[span_index(span_index(b, J, j), K, k)] += g;
                    }
                }
            }
        }
        out.rho.assign(checked_product("flat", nmb, J, K, d.Rmax), 0.0);
        for (int b = 0; b < nmb; ++b) {
            for (int j : L.categorical_index) {
                int R = d.ncat[j];
                for (int k = 0; k < K; ++k) {
                    double dd = den[span_index(span_index(b, J, j), K, k)];
                    if (dd <= 0) {
                        for (int r = 0; r < R; ++r)
                            out.rho[span_index(span_index(span_index(b, J, j), K, k), d.Rmax, r)] = 1.0 / R;
                    } else {
                        for (int r = 0; r < R; ++r)
                            out.rho[span_index(span_index(span_index(b, J, j), K, k), d.Rmax, r)] =
                                num[span_index(span_index(span_index(b, J, j), K, k), d.Rmax, r)] / dd;
                    }
                }
            }
        }
    }
    if (!L.n_gaussian) return out;
    // Gaussian-block conditional-moment M-step. The block is the product
    // factor that multiplies the categorical emission.
    const int Jg = L.n_gaussian;
    std::vector<double> nwt(checked_product("flat", T, K), 0.0);
    std::vector<Eigen::VectorXd> sumy(checked_product("flat", T, K), Eigen::VectorXd::Zero(Jg));
    EmWorkspace local_ws;
    EmWorkspace& ws = scratch ? *scratch : local_ws;
    ws.ybuf.resize(static_cast<size_t>(Jg));
    for (int i : analyzed) {
        for (int t = 0; t < T; ++t) {
            bool any = false;
            for (int g = 0; g < Jg; ++g) {
                ws.ybuf[static_cast<size_t>(g)] = d.gauss[cell(i, t, L.gaussian_index[g], T, J)];
                if (std::isfinite(ws.ybuf[static_cast<size_t>(g)])) any = true;
            }
            if (!any) continue;
            int mean_b = s.means_invariant ? 0 : t;
            int cov_b = s.covariance_invariant ? 0 : t;
            int tkey = s.covariance_invariant ? -1 : t;
            bool shared = (s.cov_model == COV_EEI || s.cov_model == COV_EEE);
            if (shared) load_sigma(current, s.cov_model, K, Jg, cov_b, 0, ws.sharedS);
            for (int k = 0; k < K; ++k) {
                double w = fb.gamma[flat_index(i, T, t, K, k)];
                if (!(w > 0)) continue;
                const double* mu = current.means.data() + span_index(span_index(mean_b, K, k), Jg, 0);
                const Eigen::MatrixXd* Sp;
                if (shared) Sp = &ws.sharedS;
                else {
                    load_sigma(current, s.cov_model, K, Jg, cov_b, k, ws.stateS);
                    Sp = &ws.stateS;
                }
                int kkey = shared ? -1 : k;
                conditional_moments(ws.ybuf.data(), mu, *Sp, Jg, ws.ey, ws.Eyy, ws.cache, tkey, kkey,
                                    regularized);
                nwt[span_index(t, K, k)] += w;
                sumy[span_index(t, K, k)] += w * ws.ey;
            }
        }
    }
    int nmb = n_mean_blocks(d, s);
    out.means.assign(checked_product("flat", nmb, K, Jg), 0.0);
    std::vector<Eigen::VectorXd> mu(checked_product("flat", nmb, K), Eigen::VectorXd::Zero(Jg));
    std::vector<double> nmu(checked_product("flat", nmb, K), 0.0);
    for (int t = 0; t < T; ++t) {
        int b = s.means_invariant ? 0 : t;
        for (int k = 0; k < K; ++k) {
            nmu[span_index(b, K, k)] += nwt[span_index(t, K, k)];
            mu[span_index(b, K, k)] += sumy[span_index(t, K, k)];
        }
    }
    const bool precision_mean = s.means_invariant && !s.covariance_invariant;
    for (int b = 0; b < nmb; ++b) {
        for (int k = 0; k < K; ++k) {
            if (nmu[span_index(b, K, k)] > 0) mu[span_index(b, K, k)] /= nmu[span_index(b, K, k)];
            if (precision_mean && nmu[span_index(b, K, k)] > 0) {
                Eigen::VectorXd solved(Jg);
                if (precision_weighted_mean(d, s, current, Jg, k, nwt, sumy, solved))
                    mu[span_index(b, K, k)] = solved;
            }
            for (int j = 0; j < Jg; ++j) out.means[span_index(span_index(b, K, k), Jg, j)] = mu[span_index(b, K, k)](j);
        }
    }
    int ncb = n_cov_blocks(d, s);
    out.cov.assign(ncb * cov_block_size(s.cov_model, K, Jg), 0.0);
    std::vector<Eigen::MatrixXd> scat(checked_product("flat", ncb, K), Eigen::MatrixXd::Zero(Jg, Jg));
    std::vector<double> nsc(checked_product("flat", ncb, K), 0.0);
    // Second pass. Center the conditional mean at the updated mean and add the
    // conditional covariance. Do not rebuild the scatter from raw moments.
    for (int i : analyzed) {
        for (int t = 0; t < T; ++t) {
            bool any = false;
            for (int g = 0; g < Jg; ++g) {
                ws.ybuf[static_cast<size_t>(g)] = d.gauss[cell(i, t, L.gaussian_index[g], T, J)];
                if (std::isfinite(ws.ybuf[static_cast<size_t>(g)])) any = true;
            }
            if (!any) continue;
            int mean_b = s.means_invariant ? 0 : t;
            int cov_b = s.covariance_invariant ? 0 : t;
            int tkey = s.covariance_invariant ? -1 : t;
            int cb = s.covariance_invariant ? 0 : t;
            bool shared_cov = (s.cov_model == COV_EEI || s.cov_model == COV_EEE);
            if (shared_cov) load_sigma(current, s.cov_model, K, Jg, cov_b, 0, ws.sharedS);
            for (int k = 0; k < K; ++k) {
                double w = fb.gamma[flat_index(i, T, t, K, k)];
                if (!(w > 0)) continue;
                const double* mu_cur = current.means.data() + span_index(span_index(mean_b, K, k), Jg, 0);
                const Eigen::MatrixXd* Sp;
                if (shared_cov) Sp = &ws.sharedS;
                else {
                    load_sigma(current, s.cov_model, K, Jg, cov_b, k, ws.stateS);
                    Sp = &ws.stateS;
                }
                int kkey = shared_cov ? -1 : k;
                conditional_moments(ws.ybuf.data(), mu_cur, *Sp, Jg, ws.ey, ws.Eyy, ws.cache, tkey, kkey,
                                    regularized);
                const Eigen::VectorXd& m = mu[span_index(mean_b, K, k)];
                Eigen::VectorXd dev = ws.ey - m;
                scat[span_index(cb, K, k)] += w * (ws.Eyy + dev * dev.transpose());
                nsc[span_index(cb, K, k)] += w;
            }
        }
    }
    bool shared = (s.cov_model == COV_EEI || s.cov_model == COV_EEE);
    bool diag = (s.cov_model == COV_EEI || s.cov_model == COV_VVI);
    for (int b = 0; b < ncb; ++b) {
        if (shared) {
            Eigen::MatrixXd acc = Eigen::MatrixXd::Zero(Jg, Jg);
            double n = 0;
            for (int k = 0; k < K; ++k) {
                acc += scat[span_index(b, K, k)];
                n += nsc[span_index(b, K, k)];
            }
            if (n > 0) acc /= n;
            if (diag) acc = Eigen::MatrixXd(acc.diagonal().asDiagonal());
            bool rid = ridge_if_needed(acc, min_eigen);
            if (rid && regularized) *regularized = true;
            store_sigma_block(out.cov, s.cov_model, K, Jg, b, 0, acc);
        } else {
            for (int k = 0; k < K; ++k) {
                Eigen::MatrixXd acc = Eigen::MatrixXd::Zero(Jg, Jg);
                if (nsc[span_index(b, K, k)] > 0) acc = scat[span_index(b, K, k)] / nsc[span_index(b, K, k)];
                else acc = Eigen::MatrixXd::Identity(Jg, Jg);
                if (diag) acc = Eigen::MatrixXd(acc.diagonal().asDiagonal());
                bool rid = ridge_if_needed(acc, min_eigen);
                if (rid && regularized) *regularized = true;
                store_sigma_block(out.cov, s.cov_model, K, Jg, b, k, acc);
            }
        }
    }
    return out;
}

void fill_counts_from_hard(const Panel& d, const Spec& s, const std::vector<int>& state,
                           const std::vector<int>& analyzed, Params& par) {
    const int K = s.K, T = d.T, J = d.J;
    std::vector<double> init(K, 0.0);
    for (int i : analyzed) init[state[wave_index(i, T, 0)]] += 1;
    double is = std::accumulate(init.begin(), init.end(), 0.0);
    par.delta.resize(K);
    for (int k = 0; k < K; ++k) par.delta[k] = (is > 0 ? init[k] / is : 1.0 / K);
    int ntb = n_tau_blocks(d, s);
    par.tau.assign(checked_product("flat", ntb, K, K), 0.0);
    if (ntb) {
        std::vector<double> num(checked_product("flat", ntb, K, K), 0.0), den(checked_product("flat", ntb, K), 0.0);
        for (int i : analyzed)
            for (int t = 0; t < T - 1; ++t) {
                int b = s.transition_equal ? 0 : t;
                int a = state[wave_index(i, T, t)], c = state[wave_index(i, T, t + 1)];
                num[span_index(span_index(b, K, a), K, c)] += 1;
                den[span_index(b, K, a)] += 1;
            }
        for (int b = 0; b < ntb; ++b)
            for (int j = 0; j < K; ++j) {
                if (den[span_index(b, K, j)] <= 0) {
                    for (int k = 0; k < K; ++k) par.tau[span_index(span_index(b, K, j), K, k)] = 1.0 / K;
                } else {
                    for (int k = 0; k < K; ++k)
                        par.tau[span_index(span_index(b, K, j), K, k)] = std::max(num[span_index(span_index(b, K, j), K, k)], 1e-3) ;
                    double sm = 0;
                    for (int k = 0; k < K; ++k) sm += par.tau[span_index(span_index(b, K, j), K, k)];
                    for (int k = 0; k < K; ++k) par.tau[span_index(span_index(b, K, j), K, k)] /= sm;
                }
            }
    }
    const IndicatorLayout L = indicator_layout(d);
    if (L.n_categorical) {
        int nmb = n_meas_blocks(d, s);
        par.rho.assign(checked_product("flat", nmb, J, K, d.Rmax), 0.0);
        std::vector<double> num(checked_product("flat", nmb, J, K, d.Rmax), 0.0),
            den(checked_product("flat", nmb, J, K), 0.0);
        for (int i : analyzed)
            for (int t = 0; t < T; ++t) {
                int b = s.measurement_invariant ? 0 : t;
                int k = state[wave_index(i, T, t)];
                for (int j : L.categorical_index) {
                    int y = d.cat[cell(i, t, j, T, J)];
                    if (y < 0) continue;
                    num[span_index(span_index(span_index(b, J, j), K, k), d.Rmax, y)] += 1;
                    den[span_index(span_index(b, J, j), K, k)] += 1;
                }
            }
        for (int b = 0; b < nmb; ++b)
            for (int j : L.categorical_index) {
                int R = d.ncat[j];
                for (int k = 0; k < K; ++k) {
                    for (int r = 0; r < R; ++r) {
                        double v = num[span_index(span_index(span_index(b, J, j), K, k), d.Rmax, r)] + 0.5;
                        double dd = den[span_index(span_index(b, J, j), K, k)] + 0.5 * R;
                        par.rho[span_index(span_index(span_index(b, J, j), K, k), d.Rmax, r)] = v / dd;
                    }
                }
            }
    }
}

Params init_deterministic(const Panel& d, const Spec& s, const std::vector<int>& analyzed) {
    const int K = s.K, T = d.T, J = d.J;
    const IndicatorLayout L = indicator_layout(d);
    std::vector<double> score(checked_product("subject-wave", d.N, T), 0.0);
    std::vector<int> hard(checked_product("subject-wave", d.N, T), 0);
    std::vector<double> vals;
    vals.reserve(analyzed.size() * T);
    for (int i : analyzed) {
        for (int t = 0; t < T; ++t) {
            double sm = 0;
            int n = 0;
            for (int j = 0; j < J; ++j) {
                if (L.family[j] == FAM_GAUSSIAN) {
                    double y = d.gauss[cell(i, t, j, T, J)];
                    if (std::isfinite(y)) {
                        sm += y;
                        ++n;
                    }
                } else {
                    int y = d.cat[cell(i, t, j, T, J)];
                    if (y >= 0) {
                        sm += y;
                        ++n;
                    }
                }
            }
            score[wave_index(i, T, t)] = n ? sm / n : 0;
            if (n) vals.push_back(score[wave_index(i, T, t)]);
        }
    }
    std::sort(vals.begin(), vals.end());
    auto quantile = [&](double p) {
        if (vals.empty()) return 0.0;
        double idx = p * (vals.size() - 1);
        int i0 = static_cast<int>(idx);
        double f = idx - i0;
        if (i0 + 1 >= static_cast<int>(vals.size())) return vals.back();
        return (1 - f) * vals[i0] + f * vals[i0 + 1];
    };
    for (int i : analyzed) {
        for (int t = 0; t < T; ++t) {
            int g = 0;
            double sc = score[wave_index(i, T, t)];
            for (int k = 1; k < K; ++k) {
                if (sc > quantile(static_cast<double>(k) / K)) g = k;
            }
            hard[wave_index(i, T, t)] = g;
        }
    }
    Params par;
    fill_counts_from_hard(d, s, hard, analyzed, par);
    if (L.n_gaussian) {
        const int Jg = L.n_gaussian;
        int nmb = n_mean_blocks(d, s);
        int ncb = n_cov_blocks(d, s);
        par.means.assign(checked_product("flat", nmb, K, Jg), 0.0);
        std::vector<double> nn(checked_product("flat", nmb, K), 0.0);
        std::vector<Eigen::VectorXd> sum(checked_product("flat", nmb, K), Eigen::VectorXd::Zero(Jg));
        std::vector<double> gn(static_cast<std::size_t>(Jg), 0.0);
        std::vector<double> gmean(static_cast<std::size_t>(Jg), 0.0);
        std::vector<double> gm2(static_cast<std::size_t>(Jg), 0.0);
        for (int i : analyzed)
            for (int t = 0; t < T; ++t) {
                int b = s.means_invariant ? 0 : t;
                int k = hard[wave_index(i, T, t)];
                bool any = false;
                for (int g = 0; g < Jg; ++g) {
                    double y = d.gauss[cell(i, t, L.gaussian_index[g], T, J)];
                    if (!std::isfinite(y)) continue;
                    sum[span_index(b, K, k)](g) += y;
                    double& count = gn[static_cast<std::size_t>(g)];
                    double& mean = gmean[static_cast<std::size_t>(g)];
                    count += 1.0;
                    double delta = y - mean;
                    mean += delta / count;
                    gm2[static_cast<std::size_t>(g)] += delta * (y - mean);
                    any = true;
                }
                if (any) nn[span_index(b, K, k)] += 1;
            }
        for (int b = 0; b < nmb; ++b)
            for (int k = 0; k < K; ++k) {
                if (nn[span_index(b, K, k)] > 0) sum[span_index(b, K, k)] /= nn[span_index(b, K, k)];
                for (int g = 0; g < Jg; ++g) par.means[span_index(span_index(b, K, k), Jg, g)] = sum[span_index(b, K, k)](g);
            }
        Eigen::VectorXd var = Eigen::VectorXd::Ones(Jg);
        for (int g = 0; g < Jg; ++g) {
            if (gn[static_cast<std::size_t>(g)] > 1.0)
                var(g) = std::max(gm2[static_cast<std::size_t>(g)] / gn[static_cast<std::size_t>(g)], 1e-3);
        }
        par.cov.assign(ncb * cov_block_size(s.cov_model, K, Jg), 0.0);
        for (int b = 0; b < ncb; ++b) {
            if (s.cov_model == COV_EEI) {
                for (int g = 0; g < Jg; ++g) par.cov[span_index(b, Jg, g)] = var(g);
            } else if (s.cov_model == COV_VVI) {
                for (int k = 0; k < K; ++k)
                    for (int g = 0; g < Jg; ++g) par.cov[span_index(span_index(b, K, k), Jg, g)] = var(g);
            } else if (s.cov_model == COV_EEE) {
                for (int a = 0; a < Jg; ++a)
                    par.cov[static_cast<std::size_t>(b) * static_cast<std::size_t>(packed_size(Jg)) +
                            packed_index(a, a)] = var(a);
            } else {
                for (int k = 0; k < K; ++k)
                    for (int a = 0; a < Jg; ++a)
                        par.cov[span_index(span_index(b, K, k), packed_size(Jg), 0) + packed_index(a, a)] =
                            var(a);
            }
        }
    }
    return par;
}

Params init_random_equi(const Panel& d, const Spec& s, const std::vector<int>& analyzed, PCG& rng) {
    const int K = s.K, T = d.T;
    std::vector<int> hard(checked_product("subject-wave", d.N, T), 0);
    std::vector<double> p(K);
    for (int i : analyzed) {
        for (int t = 0; t < T; ++t) {
            dirichlet(rng, p.data(), K);
            int arg = 0;
            double best = -1;
            for (int k = 0; k < K; ++k)
                if (p[k] > best) {
                    best = p[k];
                    arg = k;
                }
            hard[wave_index(i, T, t)] = arg;
        }
    }
    Params par;
    fill_counts_from_hard(d, s, hard, analyzed, par);
    const IndicatorLayout L = indicator_layout(d);
    if (L.n_gaussian) {
        const int Jg = L.n_gaussian;
        Params det = init_deterministic(d, s, analyzed);
        int nmb = n_mean_blocks(d, s);
        par.means.assign(checked_product("flat", nmb, K, Jg), 0.0);
        std::vector<double> nn(checked_product("flat", nmb, K), 0.0);
        for (int i : analyzed)
            for (int t = 0; t < T; ++t) {
                int b = s.means_invariant ? 0 : t;
                int k = hard[wave_index(i, T, t)];
                for (int g = 0; g < Jg; ++g) {
                    double y = d.gauss[cell(i, t, L.gaussian_index[g], T, d.J)];
                    if (!std::isfinite(y)) continue;
                    par.means[span_index(span_index(b, K, k), Jg, g)] += y;
                }
                nn[span_index(b, K, k)] += 1;
            }
        for (int b = 0; b < nmb; ++b)
            for (int k = 0; k < K; ++k) {
                double n = std::max(nn[span_index(b, K, k)], 1.0);
                for (int g = 0; g < Jg; ++g) {
                    par.means[span_index(span_index(b, K, k), Jg, g)] =
                        (nn[span_index(b, K, k)] > 0 ? par.means[span_index(span_index(b, K, k), Jg, g)] / n
                                           : det.means[span_index(span_index(b, K, k), Jg, g)]) +
                        (rng.uni() - 0.5) * 0.1;
                }
            }
        // Diagonal and shared-diagonal models keep the deterministic
        // variances. An unrestricted covariance does not. Copying that
        // diagonal into every EEE/VVV start leaves a negative within-class
        // correlation undiscovered at a 10-iteration screen, so the screen
        // ranks a slow climb to a better mode below starts that have already
        // flattened at a worse one. Centers drawn from observed rows, with a
        // random positive-definite equicorrelation, put that mode inside the
        // screen. Scales stay a fraction of the marginal variance, so the
        // draw is not a boundary start.
        const bool unrestricted = s.cov_model == COV_EEE || s.cov_model == COV_VVV;
        if (!unrestricted) {
            par.cov = det.cov;
        } else {
            std::vector<double> mcount(static_cast<std::size_t>(Jg), 0.0);
            std::vector<double> mmean_acc(static_cast<std::size_t>(Jg), 0.0);
            std::vector<double> mm2(static_cast<std::size_t>(Jg), 0.0);
            for (int i : analyzed)
                for (int t = 0; t < T; ++t)
                    for (int g = 0; g < Jg; ++g) {
                        double y = d.gauss[cell(i, t, L.gaussian_index[g], T, d.J)];
                        if (!std::isfinite(y)) continue;
                        double& count = mcount[static_cast<std::size_t>(g)];
                        double& mean = mmean_acc[static_cast<std::size_t>(g)];
                        count += 1.0;
                        double delta = y - mean;
                        mean += delta / count;
                        mm2[static_cast<std::size_t>(g)] += delta * (y - mean);
                    }
            Eigen::VectorXd sd(Jg);
            Eigen::VectorXd mmean(Jg);
            for (int g = 0; g < Jg; ++g) {
                mmean(g) = mmean_acc[static_cast<std::size_t>(g)];
                double v = mcount[static_cast<std::size_t>(g)] > 1.0
                               ? mm2[static_cast<std::size_t>(g)] / mcount[static_cast<std::size_t>(g)]
                               : 1.0;
                sd(g) = std::sqrt(std::max(v, 1e-3));
            }
            for (int b = 0; b < nmb; ++b) {
                std::vector<std::pair<int, int>> rows;
                rows.reserve(analyzed.size());
                for (int i : analyzed)
                    for (int t = 0; t < T; ++t) {
                        if (!s.means_invariant && t != b) continue;
                        bool any = false;
                        for (int g = 0; g < Jg; ++g) {
                            double y = d.gauss[cell(i, t, L.gaussian_index[g], T, d.J)];
                            if (std::isfinite(y)) any = true;
                        }
                        if (any) rows.emplace_back(i, t);
                    }
                if (static_cast<int>(rows.size()) >= K) {
                    for (int k = 0; k < K; ++k) {
                        int span = static_cast<int>(rows.size()) - k;
                        int j = k + static_cast<int>(rng.uni() * static_cast<double>(span));
                        if (j < k) j = k;
                        if (j >= static_cast<int>(rows.size())) j = static_cast<int>(rows.size()) - 1;
                        std::swap(rows[k], rows[j]);
                        int i = rows[k].first, t = rows[k].second;
                        for (int g = 0; g < Jg; ++g) {
                            double y = d.gauss[cell(i, t, L.gaussian_index[g], T, d.J)];
                            par.means[span_index(span_index(b, K, k), Jg, g)] = std::isfinite(y) ? y : mmean(g);
                        }
                    }
                }
            }
            int ncb = n_cov_blocks(d, s);
            par.cov.assign(ncb * cov_block_size(s.cov_model, K, Jg), 0.0);
            const int n_class = s.cov_model == COV_EEE ? 1 : K;
            // Equicorrelation is positive definite for rho > -1/(J-1).
            // The draw stays inside [-0.9, 0.9] and that bound.
            const double rho_lo = Jg <= 1 ? 0.0 : std::max(-0.9, -1.0 / (Jg - 1) + 0.05);
            const double rho_hi = Jg <= 1 ? 0.0 : 0.9;
            for (int b = 0; b < ncb; ++b) {
                for (int k = 0; k < n_class; ++k) {
                    double rho = rho_lo + (rho_hi - rho_lo) * rng.uni();
                    Eigen::VectorXd scale(Jg);
                    for (int g = 0; g < Jg; ++g) scale(g) = 0.3 + 0.9 * rng.uni();
                    Eigen::MatrixXd S(Jg, Jg);
                    for (int a = 0; a < Jg; ++a) {
                        S(a, a) = scale(a) * sd(a) * scale(a) * sd(a);
                        for (int c = 0; c < a; ++c) {
                            double v = rho * scale(a) * scale(c) * sd(a) * sd(c);
                            S(a, c) = S(c, a) = v;
                        }
                    }
                    store_sigma_block(par.cov, s.cov_model, K, Jg, b, k, S);
                }
            }
        }
    } else {
        // jitter rho inside the simplex
        for (size_t b = 0; b < par.rho.size();) {
            // leave as is; Dirichlet-smoothed counts already random via assignment
            break;
        }
        (void)rng;
    }
    return par;
}

// Random starts have to be able to land in a small class. A Dirichlet(1)
// label with a 0.05 mean jitter, and a copied marginal variance, stays in
// the large balanced basin: the 10-iteration screen then never sees the
// small-class mode. These starts draw the partition first (an axis tail, a
// nearest-neighbour outlier, k-means++, a rare response pattern, or an
// explicit small subset) and draw a fresh variance or item-probability
// vector for that partition. The unrestricted equicorrelation initializer
// stays available beside them; it is what reaches the replicated VVV mode.
struct PersonFeat {
    std::vector<double> g;
    std::vector<int> c;
};

std::vector<PersonFeat> person_features(const Panel& d, const Spec& s, const std::vector<int>& analyzed) {
    (void)s;
    const IndicatorLayout L = indicator_layout(d);
    const int Jg = L.n_gaussian;
    const int nc = static_cast<int>(L.categorical_index.size());
    std::vector<PersonFeat> feat(analyzed.size());
    for (size_t p = 0; p < analyzed.size(); ++p) {
        int i = analyzed[p];
        feat[p].g.assign(Jg, 0.0);
        feat[p].c.assign(nc, -1);
        std::vector<double> ng(Jg, 0.0);
        for (int t = 0; t < d.T; ++t) {
            for (int g = 0; g < Jg; ++g) {
                double y = d.gauss[cell(i, t, L.gaussian_index[g], d.T, d.J)];
                if (!std::isfinite(y)) continue;
                feat[p].g[g] += y;
                ng[g] += 1;
            }
            for (int ci = 0; ci < nc; ++ci) {
                if (feat[p].c[ci] >= 0) continue;
                int j = L.categorical_index[ci];
                int y = d.cat[cell(i, t, j, d.T, d.J)];
                if (y >= 0) feat[p].c[ci] = y;
            }
        }
        for (int g = 0; g < Jg; ++g) {
            if (ng[g] > 0) feat[p].g[g] /= ng[g];
            else feat[p].g[g] = kNaN;
        }
    }
    return feat;
}

void marginal_gaussian(const std::vector<PersonFeat>& feat, int Jg, std::vector<double>& mean,
                       std::vector<double>& sd) {
    mean.assign(Jg, 0.0);
    sd.assign(Jg, 1.0);
    std::vector<double> n(Jg, 0.0), s2(Jg, 0.0);
    for (const auto& row : feat) {
        for (int g = 0; g < Jg; ++g) {
            if (!std::isfinite(row.g[g])) continue;
            mean[g] += row.g[g];
            s2[g] += row.g[g] * row.g[g];
            n[g] += 1;
        }
    }
    for (int g = 0; g < Jg; ++g) {
        if (n[g] > 0) mean[g] /= n[g];
        double v = n[g] > 1 ? s2[g] / n[g] - mean[g] * mean[g] : 1.0;
        sd[g] = std::sqrt(std::max(v, 1e-3));
    }
}

double person_dist2(const PersonFeat& a, const PersonFeat& b, const std::vector<double>& sd) {
    double s = 0;
    for (int g = 0; g < static_cast<int>(a.g.size()); ++g) {
        if (!std::isfinite(a.g[g]) || !std::isfinite(b.g[g])) continue;
        double z = (a.g[g] - b.g[g]) / sd[g];
        s += z * z;
    }
    for (int c = 0; c < static_cast<int>(a.c.size()); ++c) {
        if (a.c[c] < 0 || b.c[c] < 0) continue;
        if (a.c[c] != b.c[c]) s += 1;
    }
    return s;
}

int pick_prop(PCG& rng, const std::vector<double>& w) {
    double sum = 0;
    for (double v : w) sum += v > 0 ? v : 0;
    int n = static_cast<int>(w.size());
    if (n <= 0) return 0;
    if (!(sum > 0)) {
        int j = static_cast<int>(rng.uni() * n);
        return std::min(std::max(j, 0), n - 1);
    }
    double u = rng.uni() * sum;
    double c = 0;
    for (int i = 0; i < n; ++i) {
        c += w[i] > 0 ? w[i] : 0;
        if (u <= c) return i;
    }
    return n - 1;
}

void kmeans_on(const std::vector<PersonFeat>& feat, const std::vector<int>& which, int K,
               const std::vector<double>& sd, PCG& rng, std::vector<int>& out_label) {
    const int n = static_cast<int>(which.size());
    out_label.assign(n, 0);
    if (n <= 0 || K <= 1) return;
    int kk = std::min(K, n);
    std::vector<int> center(kk, 0);
    center[0] = static_cast<int>(rng.uni() * n);
    if (center[0] >= n) center[0] = n - 1;
    std::vector<double> dist(n, kInf);
    for (int k = 1; k < kk; ++k) {
        for (int i = 0; i < n; ++i) {
            double d2 = person_dist2(feat[which[i]], feat[which[center[k - 1]]], sd);
            dist[i] = std::min(dist[i], d2);
        }
        center[k] = pick_prop(rng, dist);
    }
    for (int i = 0; i < n; ++i) {
        int best = 0;
        double bd = kInf;
        for (int k = 0; k < kk; ++k) {
            double d2 = person_dist2(feat[which[i]], feat[which[center[k]]], sd);
            if (d2 < bd) {
                bd = d2;
                best = k;
            }
        }
        out_label[i] = best;
    }
}

void labels_to_hard(const Panel& d, const std::vector<int>& analyzed, const std::vector<int>& label,
                    std::vector<int>& hard) {
    hard.assign(checked_product("subject-wave", d.N, d.T), 0);
    for (size_t p = 0; p < analyzed.size(); ++p) {
        int i = analyzed[p];
        for (int t = 0; t < d.T; ++t) hard[wave_index(i, d.T, t)] = label[p];
    }
}

void take_extreme(const std::vector<double>& score, int m, std::vector<int>& chosen) {
    std::vector<int> order(score.size());
    std::iota(order.begin(), order.end(), 0);
    m = std::max(1, std::min(m, static_cast<int>(score.size()) - 1));
    std::partial_sort(order.begin(), order.begin() + m, order.end(),
                      [&](int a, int b) { return score[a] > score[b]; });
    chosen.assign(order.begin(), order.begin() + m);
}

void assign_small_mask(std::vector<int>& label, const std::vector<int>& small, int K, const std::vector<PersonFeat>& feat,
                       const std::vector<double>& sd, PCG& rng) {
    const int n = static_cast<int>(label.size());
    std::fill(label.begin(), label.end(), 1);
    for (int i : small) {
        if (i >= 0 && i < n) label[i] = 0;
    }
    if (K <= 2) return;
    std::vector<int> rest;
    for (int i = 0; i < n; ++i)
        if (label[i] != 0) rest.push_back(i);
    std::vector<int> sub;
    kmeans_on(feat, rest, K - 1, sd, rng, sub);
    for (size_t r = 0; r < rest.size(); ++r) label[rest[r]] = 1 + (sub.empty() ? 0 : sub[r]);
}

void store_drawn_covariance(Params& par, const Panel& d, const Spec& s, int Jg, const std::vector<double>& sd_class,
                            const std::vector<double>& sd_pool, const std::vector<double>& ncount, PCG& rng) {
    const int K = s.K;
    int ncb = n_cov_blocks(d, s);
    par.cov.assign(ncb * cov_block_size(s.cov_model, K, Jg), 0.0);
    const bool unrestricted = s.cov_model == COV_EEE || s.cov_model == COV_VVV;
    const int n_class = (s.cov_model == COV_EEI || s.cov_model == COV_EEE) ? 1 : K;
    const double rho_lo = Jg <= 1 ? 0.0 : std::max(-0.9, -1.0 / (Jg - 1) + 0.05);
    const double rho_hi = Jg <= 1 ? 0.0 : 0.9;
    for (int b = 0; b < ncb; ++b) {
        for (int k = 0; k < n_class; ++k) {
            Eigen::VectorXd scale(Jg), dev(Jg);
            for (int g = 0; g < Jg; ++g) {
                double base = sd_pool[g];
                if (n_class == K && k < static_cast<int>(ncount.size()) && ncount[k] >= 2.0) base = sd_class[span_index(k, Jg, g)];
                scale(g) = 0.45 + rng.uni() * 1.1;
                dev(g) = std::max(base, 1e-3) * scale(g);
            }
            Eigen::MatrixXd S = Eigen::MatrixXd::Zero(Jg, Jg);
            double rho = unrestricted ? rho_lo + (rho_hi - rho_lo) * rng.uni() : 0.0;
            for (int a = 0; a < Jg; ++a) {
                S(a, a) = dev(a) * dev(a);
                if (!unrestricted) continue;
                for (int c = 0; c < a; ++c) S(a, c) = S(c, a) = rho * dev(a) * dev(c);
            }
            store_sigma_block(par.cov, s.cov_model, K, Jg, b, n_class == 1 ? 0 : k, S);
        }
    }
}

void write_means(Params& par, const Panel& d, const Spec& s, int Jg, const std::vector<double>& class_mean,
                 const std::vector<double>& ncount, const std::vector<double>& marg_mean, const std::vector<double>& marg_sd,
                 PCG& rng) {
    const int K = s.K;
    int nmb = n_mean_blocks(d, s);
    par.means.assign(checked_product("flat", nmb, K, Jg), 0.0);
    for (int k = 0; k < K; ++k) {
        double width = ncount[k] <= 3.0 ? 0.05 : 0.2;
        for (int g = 0; g < Jg; ++g) {
            double mu = ncount[k] > 0 ? class_mean[span_index(k, Jg, g)] : marg_mean[g];
            if (!std::isfinite(mu)) mu = marg_mean[g];
            mu += (rng.uni() - 0.5) * 2.0 * width * marg_sd[g];
            for (int b = 0; b < nmb; ++b) par.means[span_index(span_index(b, K, k), Jg, g)] = mu;
        }
    }
}

void class_gaussian_stats(const std::vector<PersonFeat>& feat, const std::vector<int>& label, int K, int Jg,
                          std::vector<double>& class_mean, std::vector<double>& ncount, std::vector<double>& sd_class,
                          std::vector<double>& sd_pool) {
    class_mean.assign(checked_product("flat", K, Jg), 0.0);
    ncount.assign(K, 0.0);
    sd_class.assign(checked_product("flat", K, Jg), 1.0);
    sd_pool.assign(Jg, 1.0);
    std::vector<double> sum(checked_product("flat", K, Jg), 0.0), sum2(checked_product("flat", K, Jg), 0.0),
        ng(checked_product("flat", K, Jg), 0.0);
    for (size_t p = 0; p < feat.size(); ++p) {
        int k = label[p];
        if (k < 0 || k >= K) continue;
        bool any = false;
        for (int g = 0; g < Jg; ++g) {
            if (!std::isfinite(feat[p].g[g])) continue;
            sum[span_index(k, Jg, g)] += feat[p].g[g];
            sum2[span_index(k, Jg, g)] += feat[p].g[g] * feat[p].g[g];
            ng[span_index(k, Jg, g)] += 1;
            any = true;
        }
        if (any) ncount[k] += 1;
    }
    std::vector<double> scat(Jg, 0.0), npool(Jg, 0.0);
    for (int k = 0; k < K; ++k) {
        for (int g = 0; g < Jg; ++g) {
            double n = ng[span_index(k, Jg, g)];
            double mu = n > 0 ? sum[span_index(k, Jg, g)] / n : 0.0;
            class_mean[span_index(k, Jg, g)] = mu;
            double v = n > 1 ? sum2[span_index(k, Jg, g)] / n - mu * mu : 1.0;
            if (!(v > 0)) v = 1e-3;
            sd_class[span_index(k, Jg, g)] = std::sqrt(std::max(v, 1e-3));
            scat[g] += std::max(v, 0.0) * n;
            npool[g] += n;
        }
    }
    for (int g = 0; g < Jg; ++g) {
        double v = npool[g] > 0 ? scat[g] / npool[g] : 1.0;
        sd_pool[g] = std::sqrt(std::max(v, 1e-3));
    }
}

void write_rho(Params& par, const Panel& d, const Spec& s, const std::vector<PersonFeat>& feat,
               const std::vector<int>& label, PCG& rng) {
    const IndicatorLayout L = indicator_layout(d);
    if (!L.n_categorical || par.rho.empty()) return;
    const int K = s.K;
    const int nc = static_cast<int>(L.categorical_index.size());
    int nmb = n_meas_blocks(d, s);
    for (int ci = 0; ci < nc; ++ci) {
        int j = L.categorical_index[ci];
        int R = std::max(d.ncat[j], 1);
        for (int k = 0; k < K; ++k) {
            std::vector<double> cnt(R, 0.0);
            double n = 0;
            for (size_t p = 0; p < feat.size(); ++p) {
                if (label[p] != k) continue;
                int y = feat[p].c[ci];
                if (y < 0 || y >= R) continue;
                cnt[y] += 1;
                n += 1;
            }
            std::vector<double> prob(R, 1.0 / R);
            bool pure = n >= 1;
            int mode = 0;
            for (int r = 0; r < R; ++r)
                if (cnt[r] > cnt[mode]) mode = r;
            if (n >= 1 && cnt[mode] < n) pure = false;
            if (pure && n >= 1 && R == 1) {
                prob[0] = 1;
            } else if (pure && n >= 1) {
                double eps = std::pow(10.0, -2.0 - 2.0 * rng.uni());
                eps = std::min(std::max(eps, 1e-4), 1e-2);
                for (int r = 0; r < R; ++r) prob[r] = r == mode ? (1.0 - eps) : eps / std::max(R - 1, 1);
            } else if (n > 0) {
                double w = 0.35 * rng.uni();
                std::vector<double> noise(R, 0.0);
                double ns = 0;
                for (int r = 0; r < R; ++r) {
                    noise[r] = rng.uni();
                    ns += noise[r];
                    prob[r] = (cnt[r] + 0.5) / (n + 0.5 * R);
                }
                for (int r = 0; r < R; ++r) prob[r] = (1.0 - w) * prob[r] + w * noise[r] / std::max(ns, 1e-12);
            }
            double sm = 0;
            for (int r = 0; r < R; ++r) {
                prob[r] = std::min(1.0 - 1e-6, std::max(1e-6, prob[r]));
                sm += prob[r];
            }
            for (int b = 0; b < nmb; ++b)
                for (int r = 0; r < R; ++r)
                    par.rho[span_index(span_index(span_index(b, d.J, j), K, k), d.Rmax, r)] = prob[r] / sm;
        }
    }
}

void soften_delta(Params& par, int n_people) {
    if (par.delta.empty() || n_people <= 0) return;
    double sm = 0;
    for (double& v : par.delta) {
        v = std::max(v, 0.0) * n_people + 0.5;
        sm += v;
    }
    if (sm <= 0) return;
    for (double& v : par.delta) v /= sm;
}

enum StartKind { START_TAIL = 0, START_KMEANS = 1, START_SUBSET = 2, START_PATTERN = 3, START_EQUI = 4, START_PARTIAL = 5 };

StartKind kind_for(int slot, bool unrestricted, bool categorical_only) {
    if (categorical_only) {
        // Full response patterns stay on the same slots as before. The slots
        // that used to be only k-means++ now alternate with a partial mask:
        // a class defined by a subset of items, with the other items free.
        // A pure pattern cannot propose that class when the free items are
        // interior.
        if (slot % 5 != 4) return START_PATTERN;
        int among = 0;
        for (int s = 0; s < slot; ++s)
            if (s % 5 == 4) ++among;
        return (among % 2 == 0) ? START_PARTIAL : START_KMEANS;
    }
    if (unrestricted) return (slot % 2 == 0) ? START_EQUI : START_TAIL;
    if (slot % 2 == 0) return START_TAIL;
    return (slot % 4 == 1) ? START_KMEANS : START_SUBSET;
}

int ordinal_of(int slot, StartKind want, bool unrestricted, bool categorical_only) {
    int n = 0;
    for (int s = 0; s < slot; ++s)
        if (kind_for(s, unrestricted, categorical_only) == want) ++n;
    return n;
}

Params init_random_explore(const Panel& d, const Spec& s, const std::vector<int>& analyzed, PCG& rng, int variant) {
    const int K = s.K;
    const int n = static_cast<int>(analyzed.size());
    const IndicatorLayout L = indicator_layout(d);
    const int Jg = L.n_gaussian;
    const bool unrestricted = s.cov_model == COV_EEE || s.cov_model == COV_VVV;
    const bool categorical_only = Jg == 0 && L.n_categorical > 0;
    const int slot = std::max(variant - 1, 0);
    StartKind kind = kind_for(slot, unrestricted, categorical_only);
    if (kind == START_EQUI) return init_random_equi(d, s, analyzed, rng);
    // One class has no partition to explore. The equicorrelation initializer
    // still fills means and variances; a hard-label count alone leaves those
    // empty and the Gaussian EM reads off the end.
    if (n <= 1 || K < 2) return init_random_equi(d, s, analyzed, rng);

    std::vector<PersonFeat> feat = person_features(d, s, analyzed);
    std::vector<double> marg_mean, marg_sd;
    marginal_gaussian(feat, Jg, marg_mean, marg_sd);
    std::vector<int> label(n, 0);

    if (kind == START_PATTERN) {
        const int nc = static_cast<int>(feat.empty() ? 0 : feat[0].c.size());
        std::map<std::vector<int>, std::vector<int>> groups;
        for (int i = 0; i < n; ++i) {
            if (static_cast<int>(feat[i].c.size()) != nc) continue;
            bool complete = true;
            for (int y : feat[i].c)
                if (y < 0) complete = false;
            if (!complete) continue;
            groups[feat[i].c].push_back(i);
        }
        std::vector<std::pair<int, std::vector<int>>> patterns;
        for (auto& g : groups) {
            if (g.second.empty()) continue;
            if (static_cast<int>(g.second.size()) >= n) continue;
            patterns.push_back({static_cast<int>(g.second.size()), g.second});
        }
        std::sort(patterns.begin(), patterns.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });
        int ord = ordinal_of(slot, START_PATTERN, unrestricted, categorical_only);
        if (!patterns.empty()) {
            const auto& chosen = patterns[ord % static_cast<int>(patterns.size())].second;
            assign_small_mask(label, chosen, K, feat, marg_sd, rng);
        } else {
            kind = START_SUBSET;
        }
    }
    if (kind == START_PARTIAL) {
        const int nc = static_cast<int>(feat.empty() ? 0 : feat[0].c.size());
        std::vector<std::vector<int>> masks;
        auto add_masks = [&](int msize) {
            if (msize <= 0 || msize > nc) return;
            std::vector<int> comb(msize);
            std::iota(comb.begin(), comb.end(), 0);
            while (true) {
                std::map<std::vector<int>, std::vector<int>> groups;
                for (int i = 0; i < n; ++i) {
                    std::vector<int> key(msize);
                    bool ok = true;
                    for (int t = 0; t < msize; ++t) {
                        int y = feat[i].c[comb[t]];
                        if (y < 0) {
                            ok = false;
                            break;
                        }
                        key[t] = y;
                    }
                    if (ok) groups[std::move(key)].push_back(i);
                }
                for (auto& g : groups) {
                    int sz = static_cast<int>(g.second.size());
                    if (sz >= 1 && sz <= n / 2) masks.push_back(std::move(g.second));
                }
                int t = msize - 1;
                while (t >= 0 && comb[t] == nc - msize + t) --t;
                if (t < 0) break;
                ++comb[t];
                for (int u = t + 1; u < msize; ++u) comb[u] = comb[u - 1] + 1;
            }
        };
        // Two-item masks first. A class that is extreme on only some items
        // is exactly one of these groups; one-item masks follow.
        if (nc >= 2) add_masks(2);
        add_masks(1);
        int ord = ordinal_of(slot, START_PARTIAL, unrestricted, categorical_only);
        if (!masks.empty()) {
            assign_small_mask(label, masks[ord % static_cast<int>(masks.size())], K, feat, marg_sd, rng);
        } else {
            kind = START_KMEANS;
        }
    }
    if (kind == START_TAIL && Jg > 0) {
        int ord = ordinal_of(slot, START_TAIL, unrestricted, categorical_only);
        const int sizes[] = {2, 4, 1, 3, 5, 8};
        const int n_pass = 6;
        const int wave = std::max(Jg, 1) * 2;
        int pass = ord / wave;
        int within = ord % wave;
        int m = 2;
        std::vector<double> score(n, -kInf);
        if (pass < n_pass) {
            int axis = within % Jg;
            int sign = within < Jg ? -1 : 1;
            m = std::min(sizes[pass], std::max(1, n / 3));
            for (int i = 0; i < n; ++i) {
                if (!std::isfinite(feat[i].g[axis])) continue;
                score[i] = sign * (feat[i].g[axis] - marg_mean[axis]) / marg_sd[axis];
            }
        } else {
            const int rsizes[] = {1, 2, 3, 5, 8};
            m = std::min(rsizes[ord % 5], std::max(1, n / 3));
            std::vector<double> dir(Jg, 0.0);
            for (int g = 0; g < Jg; ++g) {
                double u1 = std::max(rng.uni(), 1e-12), u2 = rng.uni();
                dir[g] = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
            }
            for (int i = 0; i < n; ++i) {
                double s = 0;
                bool any = false;
                for (int g = 0; g < Jg; ++g) {
                    if (!std::isfinite(feat[i].g[g])) continue;
                    s += dir[g] * (feat[i].g[g] - marg_mean[g]) / marg_sd[g];
                    any = true;
                }
                if (any) score[i] = s;
            }
        }
        std::vector<int> chosen;
        take_extreme(score, m, chosen);
        // A random projection uses the neighbours of its most extreme row, so a
        // tight cluster that is not aligned with one indicator stays together.
        // An axis tail keeps the m most extreme rows on that indicator.
        if (pass >= n_pass && !chosen.empty()) {
            int seed_row = chosen[0];
            std::vector<int> near(n);
            std::iota(near.begin(), near.end(), 0);
            std::sort(near.begin(), near.end(), [&](int a, int b) {
                return person_dist2(feat[a], feat[seed_row], marg_sd) <
                       person_dist2(feat[b], feat[seed_row], marg_sd);
            });
            int m2 = std::min(m, n - 1);
            chosen.assign(near.begin(), near.begin() + m2);
        }
        assign_small_mask(label, chosen, K, feat, marg_sd, rng);
    } else if (kind == START_TAIL) {
        kind = START_SUBSET;
    }
    if (kind == START_KMEANS) {
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        kmeans_on(feat, all, K, marg_sd, rng, label);
    }
    if (kind == START_SUBSET) {
        int ord = ordinal_of(slot, START_SUBSET, unrestricted, categorical_only);
        const int sizes[] = {1, 2, 3, 4, 5, 8};
        int m = std::min(sizes[ord % 6], std::max(1, n / 3));
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        for (int i = 0; i < m; ++i) {
            int j = i + static_cast<int>(rng.uni() * (n - i));
            if (j < i) j = i;
            if (j >= n) j = n - 1;
            std::swap(idx[i], idx[j]);
        }
        idx.resize(m);
        assign_small_mask(label, idx, K, feat, marg_sd, rng);
    }

    std::vector<int> hard;
    labels_to_hard(d, analyzed, label, hard);
    Params par;
    fill_counts_from_hard(d, s, hard, analyzed, par);
    soften_delta(par, n);
    if (Jg > 0) {
        std::vector<double> class_mean, ncount, sd_class, sd_pool;
        class_gaussian_stats(feat, label, K, Jg, class_mean, ncount, sd_class, sd_pool);
        write_means(par, d, s, Jg, class_mean, ncount, marg_mean, marg_sd, rng);
        store_drawn_covariance(par, d, s, Jg, sd_class, sd_pool, ncount, rng);
    }
    write_rho(par, d, s, feat, label, rng);
    return par;
}

Params init_random(const Panel& d, const Spec& s, const std::vector<int>& analyzed, PCG& rng, int variant) {
    const IndicatorLayout L = indicator_layout(d);
    const bool unrestricted = L.n_gaussian && (s.cov_model == COV_EEE || s.cov_model == COV_VVV);
    // EEE and VVV keep the equicorrelation initializer on every random start.
    // A tail start reaches a small-class mode within the ten-iteration screen
    // and then fills n_full, so the equicorrelation start that is still
    // climbing toward a higher admissible mode never continues. Diagonal and
    // categorical starts use the exploratory partitions below.
    if (unrestricted) return init_random_equi(d, s, analyzed, rng);
    return init_random_explore(d, s, analyzed, rng, variant);
}

struct OneRun {
    Params params;
    double loglik = -kInf;
    int iters = 0;
    bool converged = false;
    bool numerical_fail = false;
    bool regularized = false;
    double min_eigen = kInf;
    // True when the dense posterior was released and the best start must be recomputed.
    bool dense_dropped = false;
    // Sum of gamma over subjects and waves, one entry per class. This is what
    // the start verdict needs after gamma itself has been released.
    std::vector<double> class_weight;
    std::vector<double> trace;
    FB fb;
};

void release_dense_posterior(OneRun& run, int K) {
    if (K > 0 && !run.fb.gamma.empty()) {
        run.class_weight.assign(static_cast<size_t>(K), 0.0);
        const std::size_t n = run.fb.gamma.size() / static_cast<std::size_t>(K);
        for (std::size_t i = 0; i < n; ++i)
            for (int k = 0; k < K; ++k)
                run.class_weight[static_cast<size_t>(k)] += run.fb.gamma[span_index(i, K, k)];
    }
    const bool had_gamma = !run.fb.gamma.empty();
    run.fb.gamma.clear();
    run.fb.gamma.shrink_to_fit();
    run.fb.filtered.clear();
    run.fb.filtered.shrink_to_fit();
    run.fb.xi.clear();
    run.fb.xi.shrink_to_fit();
    run.dense_dropped = had_gamma;
}

void release_screen_posterior(OneRun& run) {
    run.fb.gamma.clear();
    run.fb.gamma.shrink_to_fit();
    run.fb.filtered.clear();
    run.fb.filtered.shrink_to_fit();
    run.fb.xi.clear();
    run.fb.xi.shrink_to_fit();
    run.dense_dropped = false;
}

// dense_tail false evaluates the last update with the forward likelihood only.
// Screening discards that posterior, and the likelihood matches the full
// forward-backward value bit for bit, including the ridge flag.
OneRun run_em(const Panel& d, const Spec& s, const Params& init, const std::vector<int>& analyzed,
              int max_iter, std::vector<DebugEmStep>* notes = nullptr, bool dense_tail = true) {
    OneRun run;
    run.params = init;
    run.min_eigen = kInf;
    double prev_ll = -kInf;
    Params prev = init;
    EmWorkspace ws;
    // These describe the retained parameters. A ridge that exists only on an
    // update the next E-step rejects is not copied onto that fit.
    bool accepted_reg = false;
    double accepted_min = kInf;
    bool pending_reg = false;
    double pending_min = kInf;
    for (int it = 1; it <= max_iter; ++it) {
        FB fb = forward_backward(d, s, run.params, analyzed, &ws);
        run.iters = it;
        run.trace.push_back(fb.loglik);
        DebugEmStep note;
        note.iteration = it;
        note.evaluated = fb.loglik;
        note.regularized_emission = fb.regularized;
        const bool nonfinite = !std::isfinite(fb.loglik);
        const bool decreased =
            it > 1 && std::isfinite(fb.loglik) && fb.loglik < prev_ll - em_decrease_allowance(prev_ll);
        if (nonfinite || decreased) {
            run.numerical_fail = nonfinite;
            run.converged = false;
            run.params = prev;
            run.loglik = prev_ll;
            if (!run.trace.empty()) run.trace.back() = prev_ll;
            run.regularized = accepted_reg;
            run.min_eigen = accepted_min;
            note.nonfinite = nonfinite;
            note.decreased = decreased;
            note.kept = prev_ll;
            if (notes) notes->push_back(note);
            break;
        }
        accepted_reg = accepted_reg || pending_reg || fb.regularized;
        if (std::isfinite(pending_min)) accepted_min = std::min(accepted_min, pending_min);
        pending_reg = false;
        pending_min = kInf;
        run.regularized = accepted_reg;
        run.min_eigen = accepted_min;
        if (it > 1 && (fb.loglik - prev_ll) / (std::abs(prev_ll) + 1e-8) < s.tol) {
            const double kept_ll = fb.loglik;
            run.loglik = kept_ll;
            run.fb = std::move(fb);
            run.converged = true;
            note.kept = kept_ll;
            if (notes) notes->push_back(note);
            break;
        }
        prev = run.params;
        prev_ll = fb.loglik;
        run.loglik = fb.loglik;
        run.fb = std::move(fb);
        bool regm = false;
        double me = kInf;
        run.params = mstep(d, s, run.params, run.fb, analyzed, &regm, &me, &ws);
        ws.invalidate_factors();
        pending_reg = regm;
        pending_min = me;
        note.regularized_mstep = regm;
        note.kept = prev_ll;
        if (notes) notes->push_back(note);
        if (it == max_iter) {
            // Evaluating the updated parameters is not another iteration.
            // A drop rejects that update and is not a converged fit.
            // Screening throws the posterior away, so that evaluation is the
            // forward likelihood only. The full posterior is kept when this
            // call is the whole fit.
            double eval_ll = -kInf;
            bool eval_reg = false;
            FB dense;
            if (dense_tail) {
                dense = forward_backward(d, s, run.params, analyzed, &ws);
                eval_ll = dense.loglik;
                eval_reg = dense.regularized;
            } else {
                LikelihoodOnly fwd = forward_likelihood(d, s, run.params, analyzed);
                eval_ll = fwd.loglik;
                eval_reg = fwd.regularized;
            }
            const bool bad2 = !std::isfinite(eval_ll) || eval_ll < prev_ll - em_decrease_allowance(prev_ll);
            if (bad2) {
                run.numerical_fail = !std::isfinite(eval_ll);
                run.converged = false;
                run.params = prev;
                run.loglik = prev_ll;
                run.regularized = accepted_reg;
                run.min_eigen = accepted_min;
                if (!run.trace.empty()) run.trace.back() = prev_ll;
                if (notes) {
                    DebugEmStep final_note;
                    final_note.iteration = it;
                    final_note.evaluated = eval_ll;
                    final_note.kept = prev_ll;
                    final_note.decreased = std::isfinite(eval_ll);
                    final_note.nonfinite = !std::isfinite(eval_ll);
                    final_note.regularized_emission = eval_reg;
                    final_note.regularized_mstep = regm;
                    notes->push_back(final_note);
                }
            } else {
                accepted_reg = accepted_reg || pending_reg || eval_reg;
                if (std::isfinite(pending_min)) accepted_min = std::min(accepted_min, pending_min);
                run.regularized = accepted_reg;
                run.min_eigen = accepted_min;
                if (dense_tail) run.fb = std::move(dense);
                run.loglik = eval_ll;
                if (!run.trace.empty()) run.trace.back() = run.loglik;
                run.converged = std::isfinite(run.loglik) &&
                                (run.loglik - prev_ll) / (std::abs(prev_ll) + 1e-8) < s.tol;
                run.iters = it;
            }
        }
    }
    return run;
}

std::vector<double> observed_gaussian_variance(const Panel& d);

int count_boundary(const Panel& d, const Spec& s, const Params& p,
                    const std::vector<double>* gauss_var = nullptr) {
    int n = 0;
    auto near = [](double v) { return v <= 1e-8 || v >= 1.0 - 1e-8; };
    // A one-state model has initial probability 1 and, when there is a
    // transition, probability 1. Those are not estimated boundaries.
    if (s.K > 1) {
        for (double v : p.delta)
            if (near(v)) ++n;
        for (double v : p.tau)
            if (near(v)) ++n;
    }
    const IndicatorLayout L = indicator_layout(d);
    if (L.n_categorical) {
        int nmb = n_meas_blocks(d, s);
        for (int b = 0; b < nmb; ++b)
            for (int j : L.categorical_index)
                for (int k = 0; k < s.K; ++k)
                    for (int r = 0; r < d.ncat[j]; ++r) {
                        double v = p.rho[span_index(span_index(span_index(b, d.J, j), s.K, k), d.Rmax, r)];
                        if (near(v)) ++n;
                    }
    }
    // A variance is a boundary when it is numerically zero relative to that
    // indicator's observed variance. An absolute cutoff would change status
    // when the indicator is rescaled. Off-diagonal zeros are structural.
    if (L.n_gaussian && !p.cov.empty()) {
        const int Jg = L.n_gaussian;
        const int packed = Jg * (Jg + 1) / 2;
        const int ncb = n_cov_blocks(d, s);
        const int need = ncb * cov_block_size(s.cov_model, s.K, Jg);
        if (static_cast<int>(p.cov.size()) < need) return n;
        const std::vector<double> owned = gauss_var ? std::vector<double>{} : observed_gaussian_variance(d);
        const std::vector<double>& ref = gauss_var ? *gauss_var : owned;
        auto floor_var = [&](double v, int g) {
            if (!(v > 0.0) || !std::isfinite(v)) return true;
            if (g < 0 || g >= static_cast<int>(ref.size()) || !(ref[g] > 0.0)) return false;
            return v / ref[g] < kRelVarFloor;
        };
        for (int b = 0; b < ncb; ++b) {
            if (s.cov_model == COV_EEI) {
                for (int j = 0; j < Jg; ++j)
                    if (floor_var(p.cov[span_index(b, Jg, j)], j)) ++n;
            } else if (s.cov_model == COV_VVI) {
                for (int k = 0; k < s.K; ++k)
                    for (int j = 0; j < Jg; ++j)
                        if (floor_var(p.cov[span_index(span_index(b, s.K, k), Jg, j)], j)) ++n;
            } else if (s.cov_model == COV_EEE) {
                const double* base = p.cov.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(packed);
                for (int a = 0; a < Jg; ++a)
                    if (floor_var(base[packed_index(a, a)], a)) ++n;
            } else {
                for (int k = 0; k < s.K; ++k) {
                    const double* pk = p.cov.data() + span_index(span_index(b, s.K, k), packed, 0);
                    for (int a = 0; a < Jg; ++a)
                        if (floor_var(pk[packed_index(a, a)], a)) ++n;
                }
            }
        }
    }
    return n;
}

std::string join_reasons(const std::vector<std::string>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ",";
        s += v[i];
    }
    return s;
}

CovarianceDiagnostic diagnose_matrix(const Eigen::MatrixXd& raw, const std::vector<double>& reference) {
    CovarianceDiagnostic out;
    const int p = static_cast<int>(raw.rows());
    Eigen::MatrixXd M = 0.5 * (raw + raw.transpose());
    out.min_eigenvalue = min_eig(M);
    out.max_eigenvalue = max_eig(M);
    bool vars_positive = p > 0;
    out.min_relative_variance = kInf;
    for (int j = 0; j < p; ++j) {
        if (!(M(j, j) > 0.0) || !std::isfinite(M(j, j))) {
            vars_positive = false;
            out.min_relative_variance = 0.0;
            continue;
        }
        double ref = M(j, j);
        if (static_cast<int>(reference.size()) == p && reference[j] > 0.0 && std::isfinite(reference[j]))
            ref = reference[j];
        out.min_relative_variance = std::min(out.min_relative_variance, M(j, j) / ref);
    }
    if (p == 0) {
        out.min_relative_variance = kNaN;
        out.level = "informational";
        out.reason = "ok";
        return out;
    }
    out.positive_definite = vars_positive && std::isfinite(out.min_eigenvalue) && out.min_eigenvalue > 0.0;
    if (vars_positive) {
        Eigen::MatrixXd R(p, p);
        for (int a = 0; a < p; ++a) {
            double sa = std::sqrt(M(a, a));
            for (int b = 0; b < p; ++b) {
                double sb = std::sqrt(M(b, b));
                R(a, b) = M(a, b) / (sa * sb);
            }
        }
        R = 0.5 * (R + R.transpose());
        out.min_correlation_eigenvalue = min_eig(R);
        double mx = max_eig(R);
        if (out.min_correlation_eigenvalue > 0.0 && std::isfinite(mx))
            out.condition_number = mx / out.min_correlation_eigenvalue;
        else
            out.condition_number = kInf;
    } else {
        out.min_correlation_eigenvalue = -kInf;
        out.condition_number = kInf;
    }
    bool corr_collapsed = vars_positive && !(out.min_correlation_eigenvalue > kCorrFloor);
    bool rel_collapsed = !(out.min_relative_variance > kRelVarFloor);
    // A non-positive variance is a collapsed component. A negative eigenvalue
    // with positive variances is not positive definite and is reported on its
    // own; the correlation eigenvalue being non-positive also marks collapse.
    out.collapsed_component = !vars_positive || rel_collapsed || corr_collapsed;
    out.ill_conditioned = out.positive_definite && !out.collapsed_component &&
                          out.condition_number > kCondWarn;
    std::vector<std::string> reasons;
    if (!out.positive_definite) reasons.push_back("not_positive_definite");
    if (out.collapsed_component) reasons.push_back("collapsed_component");
    if (out.ill_conditioned) reasons.push_back("ill_conditioned_covariance");
    if (!out.positive_definite) out.level = "blocking";
    else if (out.collapsed_component || out.ill_conditioned) out.level = "warning";
    else out.level = "informational";
    out.reason = reasons.empty() ? "ok" : join_reasons(reasons);
    return out;
}

std::vector<double> observed_gaussian_variance(const Panel& d) {
    const IndicatorLayout L = indicator_layout(d);
    std::vector<double> ref(L.n_gaussian, kNaN);
    for (int g = 0; g < L.n_gaussian; ++g) {
        int j = L.gaussian_index[g];
        std::vector<double> x;
        for (int i = 0; i < d.N; ++i)
            for (int t = 0; t < d.T; ++t) {
                double y = d.gauss[cell(i, t, j, d.T, d.J)];
                if (std::isfinite(y)) x.push_back(y);
            }
        if (x.size() < 2) continue;
        double mean = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
        double ss = 0;
        for (double v : x) ss += (v - mean) * (v - mean);
        ref[g] = ss / (static_cast<double>(x.size()) - 1.0);
    }
    return ref;
}

void annotate_covariance(const Panel& d, const Spec& s, FitResult& r,
                         const std::vector<double>* gauss_var = nullptr) {
    const IndicatorLayout L = indicator_layout(d);
    r.cov_positive_definite = true;
    r.cov_ill_conditioned = false;
    r.cov_collapsed_component = false;
    r.condition_number = kNaN;
    r.min_relative_variance = kNaN;
    if (!L.n_gaussian || r.params.cov.empty()) return;
    const std::vector<double> owned = gauss_var ? std::vector<double>{} : observed_gaussian_variance(d);
    const std::vector<double>& ref = gauss_var ? *gauss_var : owned;
    const int ncb = n_cov_blocks(d, s);
    const int Jg = L.n_gaussian;
    const bool shared = s.cov_model == COV_EEI || s.cov_model == COV_EEE;
    double worst_cond = 0;
    double worst_rel = kInf;
    bool any = false;
    for (int b = 0; b < ncb; ++b) {
        int ks = shared ? 1 : s.K;
        for (int k = 0; k < ks; ++k) {
            Eigen::MatrixXd S;
            load_sigma(r.params, s.cov_model, s.K, Jg, b, k, S);
            CovarianceDiagnostic diag = diagnose_matrix(S, ref);
            any = true;
            if (!diag.positive_definite) r.cov_positive_definite = false;
            // A ridge already replaced a collapsed matrix. The stored matrix
            // is the floored one; variance_floor is the warning, and the
            // floor's own condition number is not a second finding.
            if (!r.regularized && diag.ill_conditioned) r.cov_ill_conditioned = true;
            if (!r.regularized && diag.collapsed_component) r.cov_collapsed_component = true;
            if (std::isfinite(diag.condition_number)) worst_cond = std::max(worst_cond, diag.condition_number);
            if (std::isfinite(diag.min_relative_variance))
                worst_rel = std::min(worst_rel, diag.min_relative_variance);
        }
    }
    if (any) {
        r.condition_number = worst_cond;
        r.min_relative_variance = std::isfinite(worst_rel) ? worst_rel : kNaN;
    }
}

void fill_eligibility(FitResult& r) {
    const bool failed = r.status == STATUS_UNUSABLE;
    r.eligible_further_starts = true;
    r.eligible_assignment = !failed;
    r.eligible_transitions = !failed;
    const bool meas_ok = r.cov_positive_definite && !r.cov_collapsed_component && r.n_numerically_empty == 0;
    r.eligible_measurement = !failed && meas_ok;
    r.eligible_model_comparison = !failed && r.ic_supported && r.n_starts >= 2 && r.n_best >= 2;
    std::vector<std::string> detail;
    auto add = [&](const char* proc, const char* why) { detail.push_back(std::string(proc) + ":" + why); };
    if (!r.eligible_model_comparison) {
        if (!std::isfinite(r.loglik)) add("model_comparison", "numerical_failure");
        if (!r.converged) add("model_comparison", "not_converged");
        if (r.n_starts >= 2 && r.n_best < 2) add("model_comparison", "best_loglik_not_replicated");
        if (r.regularized) add("model_comparison", "variance_floor");
        if (!r.cov_positive_definite) add("model_comparison", "not_positive_definite");
        if (r.cov_collapsed_component) add("model_comparison", "collapsed_component");
        if (r.n_numerically_empty > 0) add("model_comparison", "numerically_empty_state");
        if (r.n_starts < 2) add("model_comparison", "replication_not_assessed");
    }
    if (!r.eligible_assignment || !r.eligible_transitions) {
        const char* procs[] = {"assignment", "transitions"};
        for (const char* proc : procs) {
            if (!std::isfinite(r.loglik)) add(proc, "numerical_failure");
            if (!r.converged) add(proc, "not_converged");
            if (r.n_starts >= 2 && r.n_best < 2) add(proc, "best_loglik_not_replicated");
            if (!r.cov_positive_definite) add(proc, "not_positive_definite");
        }
    }
    if (!r.eligible_measurement) {
        if (!std::isfinite(r.loglik)) add("measurement", "numerical_failure");
        if (!r.converged) add("measurement", "not_converged");
        if (r.n_starts >= 2 && r.n_best < 2) add("measurement", "best_loglik_not_replicated");
        if (!r.cov_positive_definite) add("measurement", "not_positive_definite");
        if (r.cov_collapsed_component) add("measurement", "collapsed_component");
        if (r.n_numerically_empty > 0) add("measurement", "numerically_empty_state");
    }
    r.eligibility_detail = join_reasons(detail);
}

void prevalences_from(const Panel& d, const Spec& s, FitResult& r, const std::vector<int>& analyzed,
                      const std::vector<double>* gauss_var = nullptr) {
    const int T = d.T, K = s.K;
    const int Na = static_cast<int>(analyzed.size());
    r.prevalence_model.assign(checked_product("flat", T, K), 0.0);
    r.prevalence_posterior.assign(checked_product("flat", T, K), 0.0);
    r.prevalence_hard.assign(checked_product("flat", T, K), 0.0);
    r.expected_membership.assign(checked_product("flat", T, K), 0.0);
    r.expected_transitions.assign(checked_product("flat", T > 1 ? T - 1 : 0, K, K), 0.0);
    r.hard_transitions.assign(checked_product("flat", T > 1 ? T - 1 : 0, K, K), 0.0);
    for (int k = 0; k < K; ++k) r.prevalence_model[k] = r.params.delta[k];
    for (int t = 0; t < T - 1; ++t) {
        int b = s.transition_equal ? 0 : t;
        for (int k = 0; k < K; ++k) {
            double sm = 0;
            for (int j = 0; j < K; ++j)
                sm += r.prevalence_model[span_index(t, K, j)] * r.params.tau[span_index(span_index(b, K, j), K, k)];
            r.prevalence_model[span_index((t + 1), K, k)] = sm;
        }
    }
    r.modal.assign(checked_product("subject-wave", d.N, T), -1);
    r.viterbi.assign(checked_product("subject-wave", d.N, T), -1);
    r.viterbi_logprob.assign(d.N, kNaN);
    for (int i : analyzed) {
        double log_joint = -kInf;
        auto path = viterbi_path(d, s, r.params, i, &log_joint);
        r.viterbi_logprob[i] = log_joint;
        for (int t = 0; t < T; ++t) {
            r.viterbi[wave_index(i, T, t)] = path[t];
            int md = 0;
            double best = -1;
            for (int k = 0; k < K; ++k) {
                double g = r.gamma[flat_index(i, T, t, K, k)];
                r.expected_membership[span_index(t, K, k)] += g;
                if (g > best) {
                    best = g;
                    md = k;
                }
            }
            r.modal[wave_index(i, T, t)] = md;
            r.prevalence_hard[span_index(t, K, md)] += 1.0;
        }
        for (int t = 0; t < T - 1; ++t) {
            int a = r.modal[wave_index(i, T, t)];
            int c = r.modal[wave_index(i, T, t + 1)];
            r.hard_transitions[span_index(span_index(t, K, a), K, c)] += 1.0;
            for (int j = 0; j < K; ++j)
                for (int k = 0; k < K; ++k)
                    r.expected_transitions[span_index(span_index(t, K, j), K, k)] +=
                        r.xi[xi_index(i, T - 1, t, K, j, k)];
        }
    }
    if (Na > 0) {
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < K; ++k) {
                r.prevalence_posterior[span_index(t, K, k)] = r.expected_membership[span_index(t, K, k)] / Na;
                r.prevalence_hard[span_index(t, K, k)] /= Na;
            }
    }
    // entropy
    double Hw = 0, Hs = 0;
    for (int i : analyzed) {
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < K; ++k) {
                double g = r.gamma[flat_index(i, T, t, K, k)];
                if (g > 0) Hw -= g * std::log(g);
            }
        for (int k = 0; k < K; ++k) {
            double g = r.gamma[flat_index(i, T, 0, K, k)];
            if (g > 0) Hs -= g * std::log(g);
        }
        for (int t = 0; t < T - 1; ++t)
            for (int j = 0; j < K; ++j) {
                double g = r.gamma[flat_index(i, T, t, K, j)];
                if (!(g > 0)) continue;
                for (int k = 0; k < K; ++k) {
                    double x = r.xi[xi_index(i, T - 1, t, K, j, k)];
                    if (x > 0) Hs -= x * std::log(x / g);
                }
            }
    }
    r.entropy_wave_raw = Hw;
    r.entropy_sequence_raw = Hs;
    double denom = (Na > 0 && K > 1) ? Na * T * std::log(static_cast<double>(K)) : 0.0;
    r.entropy_wave = (denom > 0) ? 1.0 - Hw / denom : 1.0;
    r.entropy_sequence = (denom > 0) ? 1.0 - Hs / denom : 1.0;
    r.min_occupancy = 1.0;
    r.n_numerically_empty = 0;
    r.n_empty_state_waves = 0;
    r.n_negligible_transition_rows = 0;
    for (int k = 0; k < K; ++k) {
        double mx = 0;
        for (int t = 0; t < T; ++t) {
            double count = r.expected_membership[span_index(t, K, k)];
            double prop = Na ? count / Na : 0;
            r.min_occupancy = std::min(r.min_occupancy, prop);
            mx = std::max(mx, count);
            if (count < kNumericalEmpty) r.n_empty_state_waves += 1;
        }
        // Numerical emptiness is an expected count below 1e-8, not a fraction
        // of N. A substantively small state is not this condition.
        if (mx < kNumericalEmpty) r.n_numerically_empty += 1;
    }
    if (T > 1) {
        for (int t = 0; t < T - 1; ++t) {
            for (int j = 0; j < K; ++j) {
                double row = 0;
                for (int k = 0; k < K; ++k) row += r.expected_transitions[span_index(span_index(t, K, j), K, k)];
                if (row < kNumericalEmpty) r.n_negligible_transition_rows += 1;
            }
        }
    }
    annotate_covariance(d, s, r, gauss_var);
    r.n_boundary = count_boundary(d, s, r.params, gauss_var);
    // missingness on the full panel
    const int J = d.J;
    const IndicatorLayout Lmiss = indicator_layout(d);
    r.miss_indicator_n.assign(J, 0);
    r.miss_indicator_rate.assign(J, 0);
    r.miss_wave_n.assign(T, 0);
    r.miss_wave_rate.assign(T, 0);
    std::map<std::string, int> patterns;
    r.n_complete_subjects = 0;
    r.n_monotone = 0;
    r.n_intermittent = 0;
    for (int i = 0; i < d.N; ++i) {
        bool complete = true;
        for (int t = 0; t < T; ++t) {
            bool wave_full_miss = true;
            for (int j = 0; j < J; ++j) {
                bool miss = missing_at(d, Lmiss, i, t, j);
                if (miss) {
                    r.miss_indicator_n[j] += 1;
                    complete = false;
                } else {
                    wave_full_miss = false;
                }
            }
            if (wave_full_miss) r.miss_wave_n[t] += 1;
        }
        std::string pat;
        int first_miss = -1, last_obs = -1;
        for (int t = 0; t < T; ++t) {
            bool wave_full_miss = true;
            for (int j = 0; j < J; ++j) {
                if (!missing_at(d, Lmiss, i, t, j)) wave_full_miss = false;
            }
            pat.push_back(wave_full_miss ? 'M' : 'O');
            if (!wave_full_miss) last_obs = t;
            if (wave_full_miss && first_miss < 0) first_miss = t;
        }
        // monotone: a single suffix of missing waves, or a single prefix, or one internal run
        // that continues to the end (dropout) or from the start (late entry). Intermittent otherwise.
        bool mono = true;
        int changes = 0;
        for (int t = 1; t < T; ++t)
            if (pat[t] != pat[t - 1]) ++changes;
        if (changes > 1) mono = false;
        patterns[pat] += 1;
        if (complete) r.n_complete_subjects += 1;
        else if (mono) r.n_monotone += 1;
        else r.n_intermittent += 1;
        (void)last_obs;
        (void)first_miss;
    }
    double cells = static_cast<double>(d.N) * T;
    for (int j = 0; j < J; ++j) r.miss_indicator_rate[j] = cells ? r.miss_indicator_n[j] / cells : 0;
    for (int t = 0; t < T; ++t) r.miss_wave_rate[t] = d.N ? r.miss_wave_n[t] / d.N : 0;
    for (auto& kv : patterns) {
        r.miss_pattern.push_back(kv.first);
        r.miss_pattern_n.push_back(kv.second);
    }
    // ic
    double n = static_cast<double>(r.n_analyzed);
    double p = static_cast<double>(r.n_parameters);
    r.aic = -2.0 * r.loglik + 2.0 * p;
    r.bic = -2.0 * r.loglik + p * std::log(std::max(n, 1.0));
    r.sabic = -2.0 * r.loglik + p * std::log((n + 2.0) / 24.0);
    // Three diagnostic levels. Informational quantities (expected counts,
    // per-wave emptiness, negligible transition rows, the raw eigenvalue)
    // are stored and do not change status. Warnings keep the fit usable.
    // A fit-level block is a non-finite likelihood, a failure to converge,
    // a best log-likelihood that was not replicated, or a covariance that
    // is not positive definite. That last one also makes the start
    // inadmissible. Boundary estimates do not block. An absolute eigenvalue
    // is not a cutoff.
    std::vector<std::string> blocking;
    std::vector<std::string> warning;
    if (!std::isfinite(r.loglik)) blocking.push_back("numerical_failure");
    if (!r.converged) blocking.push_back("not_converged");
    if (r.n_starts >= 2 && r.n_best < 2) blocking.push_back("best_loglik_not_replicated");
    if (!r.cov_positive_definite) blocking.push_back("not_positive_definite");
    if (r.regularized) warning.push_back("variance_floor");
    if (r.n_boundary > 0) warning.push_back("boundary_parameters");
    if (r.n_starts < 2) warning.push_back("replication_not_assessed");
    if (r.cov_ill_conditioned) warning.push_back("ill_conditioned_covariance");
    if (r.cov_collapsed_component) warning.push_back("collapsed_component");
    if (r.n_numerically_empty > 0) warning.push_back("numerically_empty_state");
    std::vector<std::string> all = blocking;
    all.insert(all.end(), warning.begin(), warning.end());
    if (!blocking.empty()) r.status = STATUS_UNUSABLE;
    else if (!warning.empty()) r.status = STATUS_WARNED;
    else r.status = STATUS_OK;
    r.status_reason = all.empty() ? "ok" : join_reasons(all);
    r.ic_supported = blocking.empty() && !r.regularized && r.cov_positive_definite &&
                     !r.cov_collapsed_component && r.n_numerically_empty == 0 &&
                     std::isfinite(r.loglik);
    r.admissible = std::isfinite(r.loglik) && r.converged && !r.regularized &&
                   r.cov_positive_definite && !r.cov_collapsed_component;
    fill_eligibility(r);
}

void apply_permutation_to_params(const Panel& d, const Spec& s, Params& p, const std::vector<int>& nto) {
    const int K = s.K;
    // nto[new] = old
    std::vector<double> delta(K);
    for (int i = 0; i < K; ++i) delta[i] = p.delta[nto[i]];
    p.delta.swap(delta);
    if (!p.tau.empty()) {
        int blocks = static_cast<int>(p.tau.size()) / checked_product("flat", K, K);
        std::vector<double> tau(p.tau.size());
        for (int b = 0; b < blocks; ++b)
            for (int i = 0; i < K; ++i)
                for (int j = 0; j < K; ++j)
                    tau[span_index(span_index(b, K, i), K, j)] = p.tau[span_index(span_index(b, K, nto[i]), K, nto[j])];
        p.tau.swap(tau);
    }
    const IndicatorLayout L = indicator_layout(d);
    if (L.n_categorical && !p.rho.empty()) {
        int nmb = static_cast<int>(p.rho.size()) / checked_product("flat", d.J, K, d.Rmax);
        std::vector<double> rho(p.rho.size());
        for (int b = 0; b < nmb; ++b)
            for (int j = 0; j < d.J; ++j)
                for (int i = 0; i < K; ++i)
                    for (int r = 0; r < d.Rmax; ++r)
                        rho[span_index(span_index(span_index(b, d.J, j), K, i), d.Rmax, r)] =
                            p.rho[span_index(span_index(span_index(b, d.J, j), K, nto[i]), d.Rmax, r)];
        p.rho.swap(rho);
    }
    if (L.n_gaussian && !p.means.empty()) {
        const int Jg = L.n_gaussian;
        int nmb = static_cast<int>(p.means.size()) / checked_product("flat", K, Jg);
        std::vector<double> means(p.means.size());
        for (int b = 0; b < nmb; ++b)
            for (int i = 0; i < K; ++i)
                for (int j = 0; j < Jg; ++j)
                    means[span_index(span_index(b, K, i), Jg, j)] = p.means[span_index(span_index(b, K, nto[i]), Jg, j)];
        p.means.swap(means);
        if (s.cov_model == COV_VVI || s.cov_model == COV_VVV) {
            int ncb = n_cov_blocks(d, s);
            int one = (s.cov_model == COV_VVI) ? Jg : packed_size(Jg);
            std::vector<double> cov(p.cov.size());
            for (int b = 0; b < ncb; ++b)
                for (int i = 0; i < K; ++i)
                    for (int t = 0; t < one; ++t)
                        cov[span_index(span_index(b, K, i), one, t)] = p.cov[span_index(span_index(b, K, nto[i]), one, t)];
            p.cov.swap(cov);
        }
    }
}

std::vector<int> canonical_order(const Panel& d, const Spec& s, const Params& p) {
    const int K = s.K;
    const IndicatorLayout L = indicator_layout(d);
    std::vector<int> idx(K);
    std::iota(idx.begin(), idx.end(), 0);
    if (static_cast<int>(p.delta.size()) < K) return idx;
    auto tie = [&](int k) {
        if (L.family.empty() || L.family[0] != FAM_GAUSSIAN) {
            int R = d.ncat.empty() ? 1 : d.ncat[0];
            return p.rho.empty() ? 0.0 : p.rho[span_index((k), d.Rmax, (R - 1))];
        }
        return p.means.empty() ? 0.0 : p.means[span_index(k, L.n_gaussian, L.gaussian_position[0])];
    };
    // Union-find on pairs inside kLabelDeltaTol. A chain is one component, so
    // the order is a strict weak order. Inside a component the profile, then
    // the original index, breaks the tie. Components themselves are ordered by
    // their largest weight, then by their smallest index.
    std::vector<int> parent(static_cast<std::size_t>(K));
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](int x) {
        int root = x;
        while (parent[static_cast<std::size_t>(root)] != root) root = parent[static_cast<std::size_t>(root)];
        while (parent[static_cast<std::size_t>(x)] != root) {
            int next = parent[static_cast<std::size_t>(x)];
            parent[static_cast<std::size_t>(x)] = root;
            x = next;
        }
        return root;
    };
    for (int a = 0; a < K; ++a) {
        for (int b = a + 1; b < K; ++b) {
            if (std::abs(p.delta[static_cast<std::size_t>(a)] - p.delta[static_cast<std::size_t>(b)]) > kLabelDeltaTol)
                continue;
            int ra = find(a);
            int rb = find(b);
            if (ra != rb) parent[static_cast<std::size_t>(rb)] = ra;
        }
    }
    std::vector<double> max_delta(static_cast<std::size_t>(K), -kInf);
    std::vector<int> min_index(static_cast<std::size_t>(K), K);
    std::vector<char> seen(static_cast<std::size_t>(K), 0);
    std::vector<int> root_of(static_cast<std::size_t>(K));
    for (int i = 0; i < K; ++i) {
        int root = find(i);
        root_of[static_cast<std::size_t>(i)] = root;
        seen[static_cast<std::size_t>(root)] = 1;
        max_delta[static_cast<std::size_t>(root)] =
            std::max(max_delta[static_cast<std::size_t>(root)], p.delta[static_cast<std::size_t>(i)]);
        min_index[static_cast<std::size_t>(root)] = std::min(min_index[static_cast<std::size_t>(root)], i);
    }
    std::vector<int> roots;
    for (int i = 0; i < K; ++i)
        if (seen[static_cast<std::size_t>(i)]) roots.push_back(i);
    std::sort(roots.begin(), roots.end(), [&](int a, int b) {
        if (max_delta[static_cast<std::size_t>(a)] != max_delta[static_cast<std::size_t>(b)])
            return max_delta[static_cast<std::size_t>(a)] > max_delta[static_cast<std::size_t>(b)];
        return min_index[static_cast<std::size_t>(a)] < min_index[static_cast<std::size_t>(b)];
    });
    std::vector<int> component_rank(static_cast<std::size_t>(K), 0);
    for (int i = 0; i < static_cast<int>(roots.size()); ++i)
        component_rank[static_cast<std::size_t>(roots[static_cast<std::size_t>(i)])] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        int ra = component_rank[static_cast<std::size_t>(root_of[static_cast<std::size_t>(a)])];
        int rb = component_rank[static_cast<std::size_t>(root_of[static_cast<std::size_t>(b)])];
        if (ra != rb) return ra < rb;
        double ta = tie(a), tb = tie(b);
        if (ta != tb) return ta > tb;
        return a < b;
    });
    return idx;
}

double meas_distance(const Panel& d, const Spec& s, const Params& a, int ka, const Params& b, int kb) {
    const IndicatorLayout L = indicator_layout(d);
    double dist = 0;
    if (L.n_categorical && !a.rho.empty() && !b.rho.empty()) {
        int na = std::max(1, static_cast<int>(a.rho.size() / checked_product("flat", d.J, s.K, d.Rmax)));
        int nb = std::max(1, static_cast<int>(b.rho.size() / checked_product("flat", d.J, s.K, d.Rmax)));
        for (int j : L.categorical_index)
            for (int r = 0; r < d.ncat[j]; ++r) {
                double sa = 0, sb = 0;
                for (int t = 0; t < na; ++t)
                    sa += a.rho[span_index(span_index(span_index(t, d.J, j), s.K, ka), d.Rmax, r)];
                for (int t = 0; t < nb; ++t)
                    sb += b.rho[span_index(span_index(span_index(t, d.J, j), s.K, kb), d.Rmax, r)];
                sa /= na;
                sb /= nb;
                dist += (sa - sb) * (sa - sb);
            }
    }
    if (L.n_gaussian && !a.means.empty() && !b.means.empty()) {
        const int Jg = L.n_gaussian;
        int na = std::max(1, static_cast<int>(a.means.size() / checked_product("flat", s.K, Jg)));
        int nb = std::max(1, static_cast<int>(b.means.size() / checked_product("flat", s.K, Jg)));
        for (int j = 0; j < Jg; ++j) {
            double sa = 0, sb = 0;
            for (int t = 0; t < na; ++t) sa += a.means[span_index(span_index(t, s.K, ka), Jg, j)];
            for (int t = 0; t < nb; ++t) sb += b.means[span_index(span_index(t, s.K, kb), Jg, j)];
            sa /= na;
            sb /= nb;
            int indicator = L.gaussian_index[j];
            double mean = 0;
            int n = 0;
            for (int i = 0; i < d.N; ++i)
                for (int t = 0; t < d.T; ++t) {
                    double y = d.gauss[cell(i, t, indicator, d.T, d.J)];
                    if (!std::isfinite(y)) continue;
                    mean += y;
                    ++n;
                }
            double sd = 0;
            if (n > 1) {
                mean /= static_cast<double>(n);
                double ss = 0;
                for (int i = 0; i < d.N; ++i)
                    for (int t = 0; t < d.T; ++t) {
                        double y = d.gauss[cell(i, t, indicator, d.T, d.J)];
                        if (std::isfinite(y)) ss += (y - mean) * (y - mean);
                    }
                sd = std::sqrt(ss / (static_cast<double>(n) - 1.0));
            }
            if (!(sd > 0.0)) continue;
            double z = (sa - sb) / sd;
            dist += z * z;
        }
    }
    return dist;
}

std::vector<int> hungarian_minimize(const std::vector<std::vector<double>>& a) {
    const int n = static_cast<int>(a.size());
    const double INF = 1e300;
    std::vector<double> u(n + 1), v(n + 1);
    std::vector<int> p(n + 1), way(n + 1);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, INF);
        std::vector<char> used(n + 1, 0);
        do {
            used[j0] = 1;
            int i0 = p[j0];
            int j1 = 0;
            double delta = INF;
            for (int j = 1; j <= n; ++j) {
                if (used[j]) continue;
                double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    std::vector<int> assign(n);
    for (int j = 1; j <= n; ++j) assign[p[j] - 1] = j - 1;
    return assign;
}

void permute_posteriors(FitResult& r, int N, int T, int K, const std::vector<int>& nto) {
    auto perm_k = [&](std::vector<double>& v, int stride_blocks) {
        if (v.empty()) return;
        std::vector<double> o(v.size());
        // layout base * K [ * K ]
        if (stride_blocks == 1) {
            for (int b = 0; b < static_cast<int>(v.size()) / K; ++b)
                for (int i = 0; i < K; ++i) o[span_index(b, K, i)] = v[span_index(b, K, nto[i])];
        } else {
            for (int b = 0; b < static_cast<int>(v.size()) / checked_product("flat", K, K); ++b)
                for (int i = 0; i < K; ++i)
                    for (int j = 0; j < K; ++j)
                        o[span_index(span_index(b, K, i), K, j)] = v[span_index(span_index(b, K, nto[i]), K, nto[j])];
        }
        v.swap(o);
    };
    perm_k(r.gamma, 1);
    perm_k(r.filtered, 1);
    perm_k(r.xi, 2);
    perm_k(r.prevalence_model, 1);
    perm_k(r.prevalence_posterior, 1);
    perm_k(r.prevalence_hard, 1);
    perm_k(r.expected_membership, 1);
    perm_k(r.expected_transitions, 2);
    perm_k(r.hard_transitions, 2);
    auto remap = [&](int old) {
        if (old < 0) return old;
        for (int i = 0; i < K; ++i)
            if (nto[i] == old) return i;
        return old;
    };
    if (r.modal.size() == checked_product("subject-wave", N, T) &&
        r.viterbi.size() == r.modal.size()) {
        for (std::size_t i = 0; i < r.modal.size(); ++i) {
            r.modal[i] = remap(r.modal[i]);
            r.viterbi[i] = remap(r.viterbi[i]);
        }
    }
    (void)N;
    (void)T;
}

struct StartVerdict {
    bool admissible = false;
    std::string level = "blocking";
    std::string reason;
    std::vector<double> class_eigen;
    double min_class_n = kNaN;
};

StartVerdict verdict_of(const Panel& d, const Spec& s, const OneRun& run,
                    const std::vector<double>* gauss_var = nullptr) {
    StartVerdict v;
    std::vector<std::string> blocking;
    std::vector<std::string> warning;
    if (!std::isfinite(run.loglik)) blocking.push_back("numerical_failure");
    if (!run.converged) blocking.push_back("not_converged");
    if (run.regularized) warning.push_back("variance_floor");
    bool pd = true;
    bool collapsed = false;
    bool ill = false;
    const IndicatorLayout L = indicator_layout(d);
    v.class_eigen.assign(s.K, kNaN);
    if (L.n_gaussian && !run.params.cov.empty()) {
        const std::vector<double> owned = gauss_var ? std::vector<double>{} : observed_gaussian_variance(d);
        const std::vector<double>& ref = gauss_var ? *gauss_var : owned;
        const int ncb = n_cov_blocks(d, s);
        const int Jg = L.n_gaussian;
        const bool shared = s.cov_model == COV_EEI || s.cov_model == COV_EEE;
        const int ks = shared ? 1 : s.K;
        for (int b = 0; b < ncb; ++b) {
            for (int k = 0; k < ks; ++k) {
                Eigen::MatrixXd S;
                load_sigma(run.params, s.cov_model, s.K, Jg, b, k, S);
                CovarianceDiagnostic diag = diagnose_matrix(S, ref);
                if (shared) {
                    for (int kk = 0; kk < s.K; ++kk) v.class_eigen[kk] = diag.min_eigenvalue;
                } else if (k < s.K) {
                    v.class_eigen[k] = diag.min_eigenvalue;
                }
                if (!diag.positive_definite) pd = false;
                // A ridge replaces the stored matrix. Collapse of that floored
                // matrix is still reported; the floor itself is variance_floor.
                if (!run.regularized && diag.collapsed_component) collapsed = true;
                if (!run.regularized && diag.ill_conditioned) ill = true;
                if (run.regularized && diag.collapsed_component) collapsed = true;
            }
        }
    }
    if (!pd) blocking.push_back("not_positive_definite");
    if (collapsed) warning.push_back("collapsed_component");
    if (ill) warning.push_back("ill_conditioned_covariance");
    if (!run.class_weight.empty()) {
        v.min_class_n = kInf;
        for (double e : run.class_weight) v.min_class_n = std::min(v.min_class_n, e);
    } else if (!run.fb.gamma.empty() && s.K > 0) {
        std::vector<double> expected(s.K, 0.0);
        const std::size_t cells = checked_product("subject-wave", d.N, d.T);
        const std::size_t have = run.fb.gamma.size() / static_cast<std::size_t>(s.K);
        const std::size_t n = std::min(cells, have);
        for (std::size_t i = 0; i < n; ++i)
            for (int k = 0; k < s.K; ++k)
                expected[k] += run.fb.gamma[span_index(i, s.K, k)];
        v.min_class_n = kInf;
        for (double e : expected) v.min_class_n = std::min(v.min_class_n, e);
    }
    v.admissible = blocking.empty() && !run.regularized && !collapsed;
    if (!blocking.empty()) v.level = "blocking";
    else if (!warning.empty()) v.level = "warning";
    else v.level = "informational";
    std::vector<std::string> all = blocking;
    all.insert(all.end(), warning.begin(), warning.end());
    v.reason = all.empty() ? "ok" : join_reasons(all);
    return v;
}

}  // namespace

double debug_last_second_forward_seconds() { return g_last_second_forward_seconds; }

#ifndef LTA_SOURCE_SHA
#define LTA_SOURCE_SHA "unset"
#endif
const char* debug_library_source_sha() { return LTA_SOURCE_SHA; }

ScaledForwardBackward scaled_forward_backward(const Panel& data, const Spec& spec, const Params& params) {
    require_panel(data);
    ScaledForwardBackward out;
    const IndicatorLayout L = indicator_layout(data);
    const int N = data.N, T = data.T, K = spec.K;
    out.gamma.assign(checked_product("posterior", N, T, K), kNaN);
    out.filtered.assign(checked_product("posterior", N, T, K), kNaN);
    out.xi.assign(checked_product("transition posterior", N, T - 1, K, K), kNaN);
    out.subject.assign(static_cast<size_t>(N), kNaN);
    const PreparedPanel prepared_scaled = prepare_data(data);
    const std::vector<int>& analyzed = prepared_scaled.analyzed;
    const ParameterPrep prep = prepare_parameters(data, L, spec, params);
    std::vector<double> ybuf(static_cast<std::size_t>(L.n_gaussian));
    FactorCache cache;
    bool regularized = false;
    std::vector<double> logb(static_cast<size_t>(K));
    std::vector<double> emis(checked_product("flat", T, K)), shift(static_cast<size_t>(T));
    std::vector<double> alpha(checked_product("flat", T, K)), beta(checked_product("flat", T, K)), scale(static_cast<size_t>(T));
    double total = 0.0;
    auto fail_subject = [&](int i) {
        out.subject[static_cast<size_t>(i)] = -kInf;
        out.loglik = -kInf;
    };
    for (int i : analyzed) {
        bool impossible = false;
        for (int t = 0; t < T; ++t) {
            emission_row(data, L, spec, params, prep, i, t, logb, ybuf.data(), &regularized, cache);
            double m = -kInf;
            for (int k = 0; k < K; ++k)
                if (std::isfinite(logb[static_cast<size_t>(k)])) m = std::max(m, logb[static_cast<size_t>(k)]);
            shift[static_cast<size_t>(t)] = m;
            if (!std::isfinite(m)) {
                impossible = true;
                break;
            }
            for (int k = 0; k < K; ++k) {
                double lb = logb[static_cast<size_t>(k)];
                emis[span_index(t, K, k)] = std::isfinite(lb) ? std::exp(lb - m) : 0.0;
            }
        }
        if (impossible) {
            fail_subject(i);
            return out;
        }
        double mass = 0.0;
        for (int k = 0; k < K; ++k) {
            double dlt = (k < static_cast<int>(params.delta.size()) && params.delta[static_cast<size_t>(k)] > 0)
                             ? params.delta[static_cast<size_t>(k)]
                             : 0.0;
            alpha[static_cast<size_t>(k)] = dlt * emis[static_cast<size_t>(k)];
            mass += alpha[static_cast<size_t>(k)];
        }
        if (!(mass > 0.0) || !std::isfinite(mass)) {
            fail_subject(i);
            return out;
        }
        scale[0] = mass;
        for (int k = 0; k < K; ++k) alpha[static_cast<size_t>(k)] /= mass;
        double ll = std::log(mass) + shift[0];
        for (int t = 1; t < T; ++t) {
            int tb = spec.transition_equal ? 0 : (t - 1);
            mass = 0.0;
            for (int k = 0; k < K; ++k) {
                double s = 0.0;
                for (int j = 0; j < K; ++j) {
                    double a = params.tau[span_index(span_index(tb, K, j), K, k)];
                    s += alpha[span_index((t - 1), K, j)] * a;
                }
                double u = s * emis[span_index(t, K, k)];
                alpha[span_index(t, K, k)] = u;
                mass += u;
            }
            if (!(mass > 0.0) || !std::isfinite(mass)) {
                fail_subject(i);
                return out;
            }
            scale[static_cast<size_t>(t)] = mass;
            for (int k = 0; k < K; ++k) alpha[span_index(t, K, k)] /= mass;
            ll += std::log(mass) + shift[static_cast<size_t>(t)];
        }
        for (int k = 0; k < K; ++k) beta[span_index((T - 1), K, k)] = 1.0;
        for (int t = T - 2; t >= 0; --t) {
            int tb = spec.transition_equal ? 0 : t;
            for (int j = 0; j < K; ++j) {
                double s = 0.0;
                for (int k = 0; k < K; ++k) {
                    double a = params.tau[span_index(span_index(tb, K, j), K, k)];
                    s += a * emis[span_index((t + 1), K, k)] * beta[span_index((t + 1), K, k)];
                }
                beta[span_index(t, K, j)] = s / scale[static_cast<size_t>(t + 1)];
            }
        }
        out.subject[static_cast<size_t>(i)] = ll;
        if (!std::isfinite(ll)) {
            fail_subject(i);
            return out;
        }
        total += ll;
        for (int t = 0; t < T; ++t) {
            double gsum = 0.0;
            for (int k = 0; k < K; ++k) {
                double g = alpha[span_index(t, K, k)] * beta[span_index(t, K, k)];
                out.gamma[flat_index(i, T, t, K, k)] = g;
                gsum += g;
            }
            if (!(gsum > 0.0)) {
                fail_subject(i);
                return out;
            }
            for (int k = 0; k < K; ++k) {
                out.filtered[flat_index(i, T, t, K, k)] = alpha[span_index(t, K, k)];
                out.gamma[flat_index(i, T, t, K, k)] /= gsum;
            }
        }
        for (int t = 0; t < T - 1; ++t) {
            int tb = spec.transition_equal ? 0 : t;
            double xsum = 0.0;
            for (int j = 0; j < K; ++j) {
                for (int k = 0; k < K; ++k) {
                    double a = params.tau[span_index(span_index(tb, K, j), K, k)];
                    double x = alpha[span_index(t, K, j)] * a * emis[span_index((t + 1), K, k)] *
                               beta[span_index((t + 1), K, k)];
                    out.xi[xi_index(i, T - 1, t, K, j, k)] = x;
                    xsum += x;
                }
            }
            if (!(xsum > 0.0)) {
                fail_subject(i);
                return out;
            }
            for (int j = 0; j < K; ++j)
                for (int k = 0; k < K; ++k)
                    out.xi[xi_index(i, T - 1, t, K, j, k)] /= xsum;
        }
    }
    out.loglik = total;
    (void)regularized;
    return out;
}

std::string covariance_name(int cov_model) {
    switch (cov_model) {
        case COV_EEI: return "EEI";
        case COV_VVI: return "VVI";
        case COV_EEE: return "EEE";
        case COV_VVV: return "VVV";
        default: return "unsupported";
    }
}

int parse_covariance_name(const std::string& name) {
    std::string s = name;
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (s == "EEI" || s == "DIAG_SHARED" || s == "1" || s == "DIAGONAL_EQUAL") return COV_EEI;
    if (s == "VVI" || s == "DIAG_STATE" || s == "2" || s == "DIAGONAL_VARYING") return COV_VVI;
    if (s == "EEE" || s == "FULL_SHARED" || s == "3" || s == "FULL_EQUAL") return COV_EEE;
    if (s == "VVV" || s == "FULL_STATE" || s == "6" || s == "FULL_VARYING") return COV_VVV;
    if (s == "4" || s == "5" || s == "EII" || s == "VII" || s == "VEI" || s == "EVI" || s == "VEE" ||
        s == "EVE" || s == "VVE" || s == "EEV" || s == "VEV" || s == "EVV" || s == "SPHERICAL") {
        throw std::invalid_argument(
            "covariance '" + name +
            "' is not in Release 1. Supported: EEI (tidyLPA 1), VVI (2), EEE (3), VVV (6). "
            "tidyLPA models 4 and 5 are not aliases of mclust models.");
    }
    throw std::invalid_argument("unknown covariance '" + name + "'");
}

int count_parameters(const Panel& data, const Spec& spec) {
    const IndicatorLayout L = indicator_layout(data);
    if (spec.K < 1) throw std::invalid_argument("K must be >= 1");
    int p = spec.K - 1;
    if (data.T >= 2) p += n_tau_blocks(data, spec) * spec.K * (spec.K - 1);
    if (L.n_categorical) {
        int nb = n_meas_blocks(data, spec);
        for (int j : L.categorical_index) p += nb * spec.K * (data.ncat[j] - 1);
    }
    if (L.n_gaussian) {
        p += n_mean_blocks(data, spec) * spec.K * L.n_gaussian;
        p += n_cov_blocks(data, spec) * cov_block_size(spec.cov_model, spec.K, L.n_gaussian);
    }
    return p;
}

BuildInfo build_info() {
    BuildInfo b;
    b.eigen_version = std::to_string(EIGEN_WORLD_VERSION) + "." +
                      std::to_string(EIGEN_MAJOR_VERSION) + "." +
                      std::to_string(EIGEN_MINOR_VERSION);
#if defined(__clang__)
    b.compiler = std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    b.compiler = std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
    b.compiler = "MSVC " + std::to_string(_MSC_VER) + " full " + std::to_string(_MSC_FULL_VER);
#else
    b.compiler = "unknown";
#endif
#ifdef __FAST_MATH__
#error "fast-math is on; this core requires the compiler default, which does not enable fast-math"
#endif
#ifdef LTA_COMPILE_FLAGS
    b.flags = LTA_COMPILE_FLAGS;
#else
    b.flags = "-O2 -fno-fast-math -ffp-contract=off -DEIGEN_DONT_PARALLELIZE -DNDEBUG -DEIGEN_NO_DEBUG";
#endif
    b.flags += " fast-math=off";
    b.cxx_standard = "17";
#ifdef EIGEN_DONT_PARALLELIZE
    b.eigen_dont_parallelize = true;
#else
    b.eigen_dont_parallelize = false;
#endif
    b.external_blas = false;
    return b;
}

PreparedPanel prepare_data(const Panel& d) {
    PreparedPanel out;
    out.layout = indicator_layout(d);
    out.N = d.N;
    out.T = d.T;
    out.J = d.J;
    out.cell_stride = static_cast<std::size_t>(d.J);
    out.subject_stride = checked_product("subject stride", d.T, d.J);
    out.gauss_pattern.assign(checked_product("subject-wave", d.N, d.T), 0);
    for (int i = 0; i < d.N; ++i) {
        bool any = false;
        for (int t = 0; t < d.T; ++t) {
            std::uint64_t mask = 0;
            for (int j = 0; j < d.J; ++j) {
                const bool gaussian = out.layout.family[j] == FAM_GAUSSIAN;
                const bool missing = gaussian ? !std::isfinite(d.gauss[cell(i, t, j, d.T, d.J)])
                                              : d.cat[cell(i, t, j, d.T, d.J)] < 0;
                if (!missing) any = true;
                if (gaussian && !missing) {
                    const int g = out.layout.gaussian_position[j];
                    if (g >= 0 && g < 64) mask |= (1ULL << static_cast<unsigned>(g));
                }
            }
            out.gauss_pattern[wave_index(i, d.T, t)] = mask;
        }
        if (any) out.analyzed.push_back(i);
        else out.excluded.push_back(i);
    }
    out.data_hash = hash_panel(d);
    std::uint64_t h = 14695981039346656037ULL;
    auto mix = [&](int v) {
        for (int k = 0; k < 4; ++k) {
            h ^= static_cast<unsigned char>((v >> (8 * k)) & 255);
            h *= 1099511628211ULL;
        }
    };
    mix(d.J);
    mix(d.T);
    mix(out.layout.measurement);
    mix(d.Rmax);
    for (int f : out.layout.family) mix(f);
    for (int nc : d.ncat) mix(nc);
    std::ostringstream os;
    os << std::hex << h;
    out.schema_hash = os.str();
    return out;
}

ParameterBlocks parameter_blocks(const Panel& data, const Spec& spec) {
    ParameterBlocks blocks;
    blocks.n_tau = n_tau_blocks(data, spec);
    blocks.n_meas = n_meas_blocks(data, spec);
    blocks.n_mean = n_mean_blocks(data, spec);
    blocks.n_cov = n_cov_blocks(data, spec);
    return blocks;
}

void require_compatible_schema(const PreparedPanel& training, const PreparedPanel& prediction) {
    if (training.schema_hash != prediction.schema_hash) {
        throw std::invalid_argument(
            "prediction schema does not match training: indicator layout, category counts, or wave count differ");
    }
}

void validate_controls(const Spec& spec) {
    if (spec.max_iter < 1) throw std::invalid_argument("max_iter must be >= 1");
    // Zero keeps every start running until max_iter. A negative or non-finite
    // value is not a stopping rule.
    if (!std::isfinite(spec.tol) || spec.tol < 0.0)
        throw std::invalid_argument("tol must be finite and non-negative");
    if (!std::isfinite(spec.replicate_tol) || spec.replicate_tol < 0.0)
        throw std::invalid_argument("replicate_tol must be finite and non-negative");
}

void require_probability(const std::vector<double>& v, std::size_t begin, int n, const std::string& what) {
    if (n < 1 || begin > v.size() || static_cast<std::size_t>(n) > v.size() - begin)
        throw std::invalid_argument(what + " is shorter than its layout");
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const double x = v[begin + static_cast<std::size_t>(i)];
        if (!std::isfinite(x) || x < 0.0)
            throw std::invalid_argument(what + " must be finite and non-negative");
        sum += x;
    }
    if (!(std::abs(sum - 1.0) <= 1e-6)) throw std::invalid_argument(what + " must sum to 1");
}

void validate_params(const Panel& data, const Spec& spec, const IndicatorLayout& layout, const Params& par) {
    if (static_cast<int>(par.delta.size()) != spec.K)
        throw std::invalid_argument("delta must have one value per state");
    require_probability(par.delta, 0, spec.K, "delta");
    const int ntb = n_tau_blocks(data, spec);
    if (ntb > 0) {
        const int need = ntb * spec.K * spec.K;
        if (static_cast<int>(par.tau.size()) != need)
            throw std::invalid_argument("tau does not match the transition layout");
        for (int b = 0; b < ntb; ++b)
            for (int from = 0; from < spec.K; ++from)
                require_probability(par.tau, span_index(span_index(b, spec.K, from), spec.K, 0), spec.K,
                                    "a transition row");
    }
    if (layout.n_categorical) {
        const int nmb = n_meas_blocks(data, spec);
        const std::size_t need = static_cast<std::size_t>(nmb) * static_cast<std::size_t>(data.J) *
                                 static_cast<std::size_t>(spec.K) * static_cast<std::size_t>(data.Rmax);
        if (par.rho.size() != need) throw std::invalid_argument("rho does not match the measurement layout");
        for (int b = 0; b < nmb; ++b)
            for (int j : layout.categorical_index)
                for (int k = 0; k < spec.K; ++k)
                    require_probability(par.rho,
                                        span_index(span_index(span_index(b, data.J, j), spec.K, k), data.Rmax, 0),
                                        data.ncat[j], "a response distribution");
    } else if (!par.rho.empty()) {
        throw std::invalid_argument("rho was supplied for a model with no categorical indicator");
    }
    if (!layout.n_gaussian) {
        if (!par.means.empty() || !par.cov.empty())
            throw std::invalid_argument("means or covariance were supplied for a model with no Gaussian indicator");
        return;
    }
    const int nmean = n_mean_blocks(data, spec);
    const int need_mean = nmean * spec.K * layout.n_gaussian;
    if (static_cast<int>(par.means.size()) != need_mean)
        throw std::invalid_argument("means do not match the Gaussian layout");
    for (double x : par.means)
        if (!std::isfinite(x)) throw std::invalid_argument("means must be finite");
    const int ncb = n_cov_blocks(data, spec);
    const int one = cov_block_size(spec.cov_model, spec.K, layout.n_gaussian);
    if (static_cast<int>(par.cov.size()) != ncb * one)
        throw std::invalid_argument("covariance does not match the Gaussian layout");
    for (double x : par.cov)
        if (!std::isfinite(x)) throw std::invalid_argument("covariance must be finite");
    const bool shared = spec.cov_model == COV_EEI || spec.cov_model == COV_EEE;
    for (int b = 0; b < ncb; ++b) {
        const int states = shared ? 1 : spec.K;
        for (int k = 0; k < states; ++k) {
            Eigen::MatrixXd sigma;
            load_sigma(par, spec.cov_model, spec.K, layout.n_gaussian, b, k, sigma);
            for (int j = 0; j < layout.n_gaussian; ++j) {
                if (!(sigma(j, j) >= 0.0))
                    throw std::invalid_argument("a variance must be finite and non-negative");
            }
        }
    }
}

double observed_loglik(const Panel& data, const Spec& spec, const Params& params) {
    const PreparedPanel prepared = prepare_data(data);
    validate_params(data, spec, prepared.layout, params);
    return forward_likelihood(data, spec, params, prepared.analyzed).loglik;
}

FixedForwardBackward fixed_forward_backward(const Panel& data, const Spec& spec, const Params& params) {
    const PreparedPanel prepared = prepare_data(data);
    validate_params(data, spec, prepared.layout, params);
    FB fb = forward_backward(data, spec, params, prepared.analyzed, nullptr);
    FixedForwardBackward out;
    out.loglik = fb.loglik;
    out.gamma = std::move(fb.gamma);
    return out;
}

std::vector<double> subject_loglik(const Panel& data, const Spec& spec, const Params& params) {
    const PreparedPanel prepared = prepare_data(data);
    validate_params(data, spec, prepared.layout, params);
    return forward_likelihood(data, spec, params, prepared.analyzed).subject;
}

Params relabel_params(const Panel& data, const Spec& spec, const Params& params,
                      const std::vector<int>& new_to_old) {
    Params p = params;
    apply_permutation_to_params(data, spec, p, new_to_old);
    return p;
}

Params align_params(const Panel& data, const Spec& spec, const Params& params,
                    std::vector<int>* order_out) {
    auto order = canonical_order(data, spec, params);
    if (order_out) *order_out = order;
    return relabel_params(data, spec, params, order);
}

Alignment align_states(const Panel& data, const Spec& spec, const Params& source,
                       const Params& reference) {
    if (spec.K < 1) throw std::invalid_argument("align_states requires K >= 1");
    Alignment out;
    out.distance =
        "squared Euclidean distance on response probabilities, plus squared "
        "differences of Gaussian means divided by the pooled n-1 standard "
        "deviation of that indicator. Covariances are not in the distance. "
        "Time-varying profiles are averaged across blocks with equal weight. "
        "The assignment is the Hungarian algorithm.";
    const int K = spec.K;
    if (K == 1) {
        out.permutation = {0};
        out.cost = meas_distance(data, spec, source, 0, reference, 0);
        out.runner_up_cost = out.cost;
        out.ambiguous = false;
        out.note = "one state";
        return out;
    }
    std::vector<std::vector<double>> cost(K, std::vector<double>(K));
    for (int s = 0; s < K; ++s)
        for (int r = 0; r < K; ++r)
            cost[s][r] = meas_distance(data, spec, source, s, reference, r);
    const std::vector<int> assign = hungarian_minimize(cost);
    out.permutation.assign(K, 0);
    out.cost = 0;
    for (int s = 0; s < K; ++s) {
        out.permutation[assign[s]] = s;
        out.cost += cost[s][assign[s]];
    }
    double second = kInf;
    for (int s = 0; s < K; ++s) {
        auto held = cost;
        held[s][assign[s]] = 1e300;
        const std::vector<int> alt = hungarian_minimize(held);
        double alt_cost = 0;
        bool feasible = true;
        for (int i = 0; i < K; ++i) {
            if (!(cost[i][alt[i]] < 1e299)) feasible = false;
            alt_cost += cost[i][alt[i]];
        }
        if (feasible) second = std::min(second, alt_cost);
    }
    out.runner_up_cost = second;
    const double gap = second - out.cost;
    out.ambiguous = !(gap > 1e-4 * (1.0 + std::abs(out.cost)));
    out.note = out.ambiguous ? "best and second-best measurement-profile assignments are close"
                             : "unique measurement-profile assignment";
    return out;
}

std::vector<int> match_states(const Panel& data, const Spec& spec, const Params& source,
                              const Params& target) {
    return align_states(data, spec, source, target).permutation;
}

CovarianceDiagnostic diagnose_covariance(const std::vector<double>& sigma, int p,
                                         const std::vector<double>& reference_scale) {
    if (p < 1) throw std::invalid_argument("diagnose_covariance requires p >= 1");
    if (static_cast<int>(sigma.size()) != p * p) {
        throw std::invalid_argument("diagnose_covariance expected a row-major p by p matrix");
    }
    Eigen::MatrixXd S(p, p);
    for (int a = 0; a < p; ++a)
        for (int b = 0; b < p; ++b) S(a, b) = sigma[span_index(a, p, b)];
    return diagnose_matrix(S, reference_scale);
}

FitResult evaluate(const Panel& data, const Spec& spec, const Params& params) {
    require_panel(data);
    if (spec.K < 1) throw std::invalid_argument("n_states must be >= 1");
    const IndicatorLayout layout = indicator_layout(data);
    std::vector<int> analyzed, excluded;
    for (int i = 0; i < data.N; ++i) {
        if (subject_has_data(data, layout, i)) analyzed.push_back(i);
        else excluded.push_back(i);
    }
    if (analyzed.empty()) {
        throw std::invalid_argument(
            "every subject has zero observed indicators; nothing to evaluate");
    }
    Params used = params;
    std::vector<int> order(spec.K);
    std::iota(order.begin(), order.end(), 0);
    if (spec.align_labels && spec.K > 1) {
        order = canonical_order(data, spec, used);
        used = relabel_params(data, spec, used, order);
    }
    FB fb = forward_backward(data, spec, used, analyzed, nullptr);
    FitResult r;
    r.engine = "cpp-em";
    r.schema_version = 2;
    r.interval_type = "none";
    r.parameter_uncertainty = "none";
    r.state_uncertainty = "plugin_posterior";
    r.loglik = fb.loglik;
    r.converged = std::isfinite(fb.loglik);
    r.regularized = fb.regularized;
    r.n_iter = 0;
    r.n_starts = 1;
    r.n_full = 1;
    r.n_converged_starts = r.converged ? 1 : 0;
    r.n_best = r.converged ? 1 : 0;
    r.n_subjects = data.N;
    r.n_analyzed = static_cast<int>(analyzed.size());
    r.n_excluded = static_cast<int>(excluded.size());
    r.excluded_subjects = excluded;
    r.n_waves = data.T;
    r.n_indicators = data.J;
    r.K = spec.K;
    r.params = used;
    r.gamma = fb.gamma;
    r.filtered = fb.filtered;
    r.xi = fb.xi;
    r.label_order = order;
    r.measurement = layout.measurement;
    r.family = layout.family;
    r.gaussian_index = layout.gaussian_index;
    r.categorical_index = layout.categorical_index;
    r.n_gaussian = layout.n_gaussian;
    r.n_categorical = layout.n_categorical;
    r.cov_model = spec.cov_model;
    r.covariance_name = layout.n_gaussian ? covariance_name(spec.cov_model) : "";
    r.measurement_invariant = spec.measurement_invariant;
    r.means_invariant = spec.means_invariant;
    r.covariance_invariant = spec.covariance_invariant;
    r.transition_equal = spec.transition_equal;
    r.n_tau_blocks = n_tau_blocks(data, spec);
    r.n_meas_blocks = n_meas_blocks(data, spec);
    r.n_mean_blocks = n_mean_blocks(data, spec);
    r.n_cov_blocks = n_cov_blocks(data, spec);
    r.Rmax = data.Rmax;
    r.n_parameters = count_parameters(data, spec);
    r.data_hash = hash_panel(data);
    r.seed = spec.seed;
    r.n_threads = 1;
    r.tol = spec.tol;
    r.replicate_tol = spec.replicate_tol;
    double mine = kInf;
    if (layout.n_gaussian && !used.cov.empty()) {
        const bool shared = spec.cov_model == COV_EEI || spec.cov_model == COV_EEE;
        for (int b = 0; b < r.n_cov_blocks; ++b) {
            int ks = shared ? 1 : spec.K;
            for (int k = 0; k < ks; ++k) {
                Eigen::MatrixXd S;
                load_sigma(used, spec.cov_model, spec.K, layout.n_gaussian, b, k, S);
                mine = std::min(mine, min_eig(S));
            }
        }
    }
    r.min_eigenvalue = std::isfinite(mine) ? mine : kNaN;
    const std::vector<double> gauss_var =
        layout.n_gaussian ? observed_gaussian_variance(data) : std::vector<double>{};
    prevalences_from(data, spec, r, analyzed, layout.n_gaussian ? &gauss_var : nullptr);
    return r;
}

void permute_rejected_records(const Panel& data, const Spec& spec, FitResult& fit,
                              const std::vector<int>& nto) {
    const int nrej = static_cast<int>(fit.rejected_loglik.size());
    if (nrej <= 0 || spec.K < 1) return;
    const int K = spec.K;
    auto permute_classes = [&](std::vector<double>& v, const char* what) {
        if (v.empty()) return;
        if (static_cast<int>(v.size()) % nrej != 0) {
            throw std::invalid_argument(std::string(what) + " is not one record per rejected start");
        }
        const int stride = static_cast<int>(v.size()) / nrej;
        if (stride != K) {
            throw std::invalid_argument(std::string(what) + " is not one value per class");
        }
        for (int rec = 0; rec < nrej; ++rec) {
            const int base = rec * stride;
            std::vector<double> next(static_cast<size_t>(stride));
            for (int i = 0; i < K; ++i) next[static_cast<size_t>(i)] = v[static_cast<size_t>(base + nto[i])];
            for (int i = 0; i < K; ++i) v[static_cast<size_t>(base + i)] = next[static_cast<size_t>(i)];
        }
    };
    permute_classes(fit.rejected_delta, "rejected_delta");
    permute_classes(fit.rejected_class_eigen, "rejected_class_eigen");
    if (fit.rejected_means.empty() && fit.rejected_cov.empty()) return;
    if (!fit.rejected_means.empty() && static_cast<int>(fit.rejected_means.size()) % nrej != 0) {
        throw std::invalid_argument("rejected_means is not one record per rejected start");
    }
    if (!fit.rejected_cov.empty() && static_cast<int>(fit.rejected_cov.size()) % nrej != 0) {
        throw std::invalid_argument("rejected_cov is not one record per rejected start");
    }
    const int mean_stride =
        fit.rejected_means.empty() ? 0 : static_cast<int>(fit.rejected_means.size()) / nrej;
    const int cov_stride = fit.rejected_cov.empty() ? 0 : static_cast<int>(fit.rejected_cov.size()) / nrej;
    for (int rec = 0; rec < nrej; ++rec) {
        Params chunk;
        chunk.delta.assign(static_cast<size_t>(K), 0.0);
        if (mean_stride > 0) {
            chunk.means.assign(fit.rejected_means.begin() + rec * mean_stride,
                               fit.rejected_means.begin() + (rec + 1) * mean_stride);
        }
        if (cov_stride > 0) {
            chunk.cov.assign(fit.rejected_cov.begin() + rec * cov_stride,
                             fit.rejected_cov.begin() + (rec + 1) * cov_stride);
        }
        apply_permutation_to_params(data, spec, chunk, nto);
        if (mean_stride > 0) {
            std::copy(chunk.means.begin(), chunk.means.end(), fit.rejected_means.begin() + rec * mean_stride);
        }
        if (cov_stride > 0) {
            std::copy(chunk.cov.begin(), chunk.cov.end(), fit.rejected_cov.begin() + rec * cov_stride);
        }
    }
}

FitResult relabel_fit(const Panel& data, const Spec& spec, FitResult fit,
                      const std::vector<int>& new_to_old) {
    if (static_cast<int>(new_to_old.size()) != spec.K) {
        throw std::invalid_argument("relabel permutation must have length K");
    }
    std::vector<int> seen(spec.K, 0);
    for (int v : new_to_old) {
        if (v < 0 || v >= spec.K || seen[v]) throw std::invalid_argument("relabel is not a permutation");
        seen[v] = 1;
    }
    apply_permutation_to_params(data, spec, fit.params, new_to_old);
    permute_rejected_records(data, spec, fit, new_to_old);
    permute_posteriors(fit, data.N, data.T, spec.K, new_to_old);
    // compose label map: previous new_to_old then this one
    if (fit.label_order.size() == static_cast<size_t>(spec.K)) {
        std::vector<int> comp(spec.K);
        for (int i = 0; i < spec.K; ++i) comp[i] = fit.label_order[new_to_old[i]];
        fit.label_order.swap(comp);
    } else {
        fit.label_order = new_to_old;
    }
    return fit;
}

FitResult align_fit(const Panel& data, const Spec& spec, FitResult fit, const Params& reference,
                    Alignment* info) {
    Alignment al = align_states(data, spec, fit.params, reference);
    if (info) *info = al;
    return relabel_fit(data, spec, std::move(fit), al.permutation);
}

namespace {

int& creation_fail_after_slot() {
    static int value = -1;
    return value;
}

// Per calling thread. Bootstrap draws fit on worker threads, and a shared
// record would let one draw replace another's timing, or the caller's.
thread_local FitPhaseTiming g_phase;

// Persistent pool. Workers block until a run() publishes an epoch, then pull
// indices from an atomic counter and write into caller-owned slots. The
// destructor joins every thread that was started, including when the
// constructor throws after starting only some of them.
class WorkerPool {
   public:
    struct Load {
        int jobs = 0;
        double busy_seconds = 0;
    };

    explicit WorkerPool(int nthreads)
        : nthreads_(std::max(1, nthreads)), load_(static_cast<size_t>(nthreads_)) {
        if (nthreads_ <= 1) return;
        workers_.reserve(static_cast<size_t>(nthreads_));
        try {
            for (int t = 0; t < nthreads_; ++t) {
                if (creation_fail_after_slot() >= 0 && t >= creation_fail_after_slot()) {
                    throw std::runtime_error("injected thread creation failure");
                }
                workers_.emplace_back([this, t] { worker_loop(t); });
            }
        } catch (...) {
            shutdown();
            throw;
        }
    }

    ~WorkerPool() { shutdown(); }

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    const std::vector<Load>& loads() const { return load_; }

    template <class Fn>
    void run(int n, Fn&& fn, void (*interrupt)()) {
        if (n <= 0) return;
        for (Load& slot : load_) {
            slot.jobs = 0;
            slot.busy_seconds = 0;
        }
        if (workers_.empty()) {
            for (int i = 0; i < n; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                fn(i);
                load_[0].busy_seconds +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                load_[0].jobs += 1;
                if (interrupt) interrupt();
            }
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            job_ = std::function<void(int)>(std::forward<Fn>(fn));
            n_ = n;
            next_.store(0, std::memory_order_relaxed);
            done_.store(0, std::memory_order_relaxed);
            stop_.store(false, std::memory_order_relaxed);
            errors_.assign(static_cast<size_t>(n), std::exception_ptr{});
            parked_ = 0;
            ++epoch_;
        }
        cv_work_.notify_all();

        std::exception_ptr interrupt_error;
        int waved = 0;
        const int wave = nthreads_;
        while (waved < n && !interrupt_error) {
            const int target = std::min(n, waved + wave);
            {
                std::unique_lock<std::mutex> lock(progress_mu_);
                cv_progress_.wait(lock, [&] {
                    return done_.load(std::memory_order_acquire) >= target ||
                           stop_.load(std::memory_order_acquire);
                });
            }
            if (stop_.load(std::memory_order_acquire) &&
                done_.load(std::memory_order_acquire) < target) {
                break;
            }
            try {
                if (interrupt) interrupt();
            } catch (...) {
                stop_.store(true, std::memory_order_release);
                interrupt_error = std::current_exception();
                break;
            }
            waved = target;
        }
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_park_.wait(lock, [&] { return parked_ == nthreads_; });
        }
        if (interrupt_error) std::rethrow_exception(interrupt_error);
        for (int i = 0; i < n; ++i) {
            if (errors_[static_cast<size_t>(i)]) std::rethrow_exception(errors_[static_cast<size_t>(i)]);
        }
    }

   private:
    void worker_loop(int id) {
        int seen = 0;
        while (true) {
            std::function<void(int)> job;
            int n = 0;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_work_.wait(lock, [&] { return shutting_down_ || epoch_ != seen; });
                if (shutting_down_) return;
                seen = epoch_;
                job = job_;
                n = n_;
            }
            while (!stop_.load(std::memory_order_acquire)) {
                const int i = next_.fetch_add(1, std::memory_order_relaxed);
                if (i >= n) break;
                const auto t0 = std::chrono::steady_clock::now();
                try {
                    job(i);
                } catch (...) {
                    errors_[static_cast<size_t>(i)] = std::current_exception();
                }
                load_[static_cast<size_t>(id)].busy_seconds +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                load_[static_cast<size_t>(id)].jobs += 1;
                done_.fetch_add(1, std::memory_order_release);
                {
                    std::lock_guard<std::mutex> lock(progress_mu_);
                }
                cv_progress_.notify_one();
            }
            {
                std::lock_guard<std::mutex> lock(mu_);
                ++parked_;
                if (parked_ == nthreads_) cv_park_.notify_one();
            }
        }
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (shut_down_called_) return;
            shut_down_called_ = true;
            shutting_down_ = true;
            stop_.store(true, std::memory_order_release);
        }
        cv_work_.notify_all();
        cv_progress_.notify_all();
        for (auto& th : workers_) {
            if (th.joinable()) th.join();
        }
        workers_.clear();
    }

    int nthreads_ = 1;
    std::vector<Load> load_;
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::mutex progress_mu_;
    std::condition_variable cv_work_;
    std::condition_variable cv_progress_;
    std::condition_variable cv_park_;
    std::function<void(int)> job_;
    int n_ = 0;
    int epoch_ = 0;
    int parked_ = 0;
    bool shutting_down_ = false;
    bool shut_down_called_ = false;
    std::atomic<int> next_{0};
    std::atomic<int> done_{0};
    std::atomic<bool> stop_{false};
    std::vector<std::exception_ptr> errors_;
};

}  // namespace

bool debug_likelihood_matches_full(const Panel& data, const Spec& spec, const Params& params) {
    const IndicatorLayout layout = indicator_layout(data);
    std::vector<int> analyzed;
    for (int i = 0; i < data.N; ++i)
        if (subject_has_data(data, layout, i)) analyzed.push_back(i);
    const FB full = forward_backward(data, spec, params, analyzed, nullptr);
    const LikelihoodOnly fwd = forward_likelihood(data, spec, params, analyzed);
    if (!same_bits(full.loglik, fwd.loglik)) return false;
    if (full.regularized != fwd.regularized) return false;
    if (full.subject.size() != fwd.subject.size()) return false;
    for (size_t i = 0; i < full.subject.size(); ++i)
        if (!same_bits(full.subject[i], fwd.subject[i])) return false;
    return true;
}

DebugEmRun debug_em_from_params(const Panel& data, const Spec& spec, const Params& init, int max_iter) {
    if (max_iter < 1) throw std::invalid_argument("max_iter must be >= 1");
    require_panel(data);
    const IndicatorLayout layout = indicator_layout(data);
    validate_params(data, spec, layout, init);
    std::vector<int> analyzed;
    for (int i = 0; i < data.N; ++i)
        if (subject_has_data(data, layout, i)) analyzed.push_back(i);
    DebugEmRun out;
    OneRun run = run_em(data, spec, init, analyzed, max_iter, &out.steps);
    out.params = std::move(run.params);
    out.loglik = run.loglik;
    out.converged = run.converged;
    out.regularized = run.regularized;
    return out;
}

LogPosterior debug_log_posterior(const Panel& data, const Spec& spec, const Params& params) {
    require_panel(data);
    const IndicatorLayout layout = indicator_layout(data);
    std::vector<int> analyzed;
    for (int i = 0; i < data.N; ++i)
        if (subject_has_data(data, layout, i)) analyzed.push_back(i);
    FB fb = forward_backward(data, spec, params, analyzed, nullptr);
    LogPosterior out;
    out.loglik = fb.loglik;
    out.gamma = std::move(fb.gamma);
    out.xi = std::move(fb.xi);
    return out;
}

void debug_set_thread_creation_fail_after(int fail_after) { creation_fail_after_slot() = fail_after; }

void debug_parallel_for(int n, int nthreads, void (*job)(int, void*), void* ctx, void (*interrupt)()) {
    if (n < 0 || nthreads < 1 || (n > 0 && job == nullptr)) {
        throw std::invalid_argument("debug_parallel_for range");
    }
    WorkerPool pool(nthreads);
    pool.run(n, [&](int i) { job(i, ctx); }, interrupt);
}

FitPhaseTiming last_fit_phase_timing() { return g_phase; }

std::uint64_t hash_params(const Params& p) {
    std::uint64_t h = 14695981039346656037ULL;
    auto mix = [&](const std::vector<double>& v) {
        for (double x : v) {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &x, sizeof(bits));
            h ^= bits;
            h *= 1099511628211ULL;
        }
        h ^= 0xFF;
        h *= 1099511628211ULL;
    };
    mix(p.delta);
    mix(p.tau);
    mix(p.rho);
    mix(p.means);
    mix(p.cov);
    return h;
}

// has_start runs only the supplied parameters. append_supplied_start keeps
// the scheduled search and adds that vector as one more candidate.
int start_count(const Spec& s) {
    const int scheduled = s.has_start ? 1 : s.n_starts;
    if (s.append_supplied_start && !s.has_start) return scheduled + 1;
    return scheduled;
}

// Start 0 is deterministic only when start_variant_offset is 0. A positive
// offset, which add_starts sets to the previous n_starts, schedules n_starts
// random initializers at variants offset, offset+1, ...
std::vector<Params> schedule_starts(const Panel& data, const Spec& s, const std::vector<int>& analyzed) {
    const int nstarts = start_count(s);
    std::vector<Params> inits(static_cast<size_t>(std::max(nstarts, 0)));
    if (nstarts < 1) return inits;
    if (s.has_start) {
        inits[0] = s.start;
        return inits;
    }
    const int scheduled = s.n_starts;
    const bool random_only = s.start_variant_offset > 0;
    int first_random = 0;
    if (!random_only) {
        inits[0] = init_deterministic(data, s, analyzed);
        first_random = 1;
    }
    for (int id = first_random; id < scheduled; ++id) {
        const int variant = random_only ? s.start_variant_offset + id : id;
        PCG rng(mix_seed(s.seed, static_cast<std::uint64_t>(variant)));
        inits[static_cast<size_t>(id)] = init_random(data, s, analyzed, rng, variant);
    }
    if (s.append_supplied_start) inits[static_cast<size_t>(scheduled)] = s.start;
    return inits;
}

std::vector<std::uint64_t> debug_start_hashes(const Panel& data, const Spec& spec) {
    if (spec.n_starts < 1 && !spec.has_start) throw std::invalid_argument("n_starts must be >= 1");
    const PreparedPanel prepared = prepare_data(data);
    auto inits = schedule_starts(data, spec, prepared.analyzed);
    std::vector<std::uint64_t> hashes;
    hashes.reserve(inits.size());
    for (const Params& p : inits) hashes.push_back(hash_params(p));
    return hashes;
}

FitResult fit(const Panel& data, const Spec& spec, void (*interrupt)()) {
    if (spec.K < 1) throw std::invalid_argument("n_states must be >= 1");
    if (spec.n_starts < 1) throw std::invalid_argument("n_starts must be >= 1");
    if (spec.n_threads < 1) throw std::invalid_argument("n_threads must be >= 1");
    validate_controls(spec);
    const PreparedPanel prepared = prepare_data(data);
    const IndicatorLayout layout = prepared.layout;
    if (layout.n_gaussian && (spec.cov_model < COV_EEI || spec.cov_model > COV_VVV)) {
        throw std::invalid_argument(
            "a Gaussian indicator block requires covariance EEI, VVI, EEE or VVV");
    }
    const std::vector<int>& analyzed = prepared.analyzed;
    const std::vector<int>& excluded = prepared.excluded;
    if (analyzed.empty()) {
        throw std::invalid_argument(
            "every subject has zero observed indicators; nothing to estimate. "
            "Exclusions are not applied silently — the call is refused.");
    }
    Spec s = spec;
    if (s.n_full < 1) s.n_full = 1;
    if (s.has_start || s.append_supplied_start) validate_params(data, s, layout, s.start);
    const int nstarts = start_count(s);
    if (s.n_full > nstarts) s.n_full = nstarts;
    std::vector<Params> inits = schedule_starts(data, s, analyzed);
    std::vector<OneRun> runs(nstarts);
    int screen = s.screen_iter;
    if (nstarts == 1 || screen <= 0 || screen >= s.max_iter) screen = s.max_iter;
    const bool separate = screen < s.max_iter;
    // One pool for the screen and, after the barrier below, the continuation.
    // Slots are the start ids handed out by the pool. Seeds were drawn above
    // from those ids, on this thread.
    const int pool_n = std::min(s.n_threads, nstarts);
    WorkerPool pool(pool_n);
    const auto screen_t0 = std::chrono::steady_clock::now();
    pool.run(
        nstarts,
        [&](int id) {
            runs[id] = run_em(data, s, inits[id], analyzed, screen, nullptr, !separate);
            if (separate) release_screen_posterior(runs[id]);
            else release_dense_posterior(runs[id], s.K);
        },
        interrupt);
    const double screen_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - screen_t0).count();
    double continuation_ms = 0;
    std::vector<int> order(nstarts);
    std::iota(order.begin(), order.end(), 0);
    // A start that has already been ridged is inadmissible. Ranking it above
    // a clean climb fills n_full with points the selector will then reject,
    // and the admissible mode never gets past the screen. Clean finite
    // likelihoods go first, then ridged ones, then numerical failures.
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        auto tier = [&](int id) {
            if (runs[id].numerical_fail || !std::isfinite(runs[id].loglik)) return 2;
            if (runs[id].regularized) return 1;
            return 0;
        };
        int ta = tier(a), tb = tier(b);
        if (ta != tb) return ta < tb;
        double la = std::isfinite(runs[a].loglik) ? runs[a].loglik : -kInf;
        double lb = std::isfinite(runs[b].loglik) ? runs[b].loglik : -kInf;
        if (la != lb) return la > lb;
        return a < b;
    });
    std::vector<int> full_ids;
    if (screen < s.max_iter) {
        for (int i = 0; i < nstarts && static_cast<int>(full_ids.size()) < s.n_full; ++i) {
            if (!runs[order[i]].numerical_fail) full_ids.push_back(order[i]);
        }
        if (full_ids.empty() && nstarts) full_ids.push_back(order[0]);
        std::vector<char> keep(static_cast<size_t>(nstarts), 0);
        for (int id : full_ids) keep[static_cast<size_t>(id)] = 1;
        for (int id = 0; id < nstarts; ++id) {
            if (keep[static_cast<size_t>(id)]) continue;
            runs[id].params = Params();
            runs[id].trace.clear();
            runs[id].trace.shrink_to_fit();
        }
        // Full barrier: screening has finished, and only then do continuations start.
        std::vector<OneRun> cont(full_ids.size());
        const auto cont_t0 = std::chrono::steady_clock::now();
        pool.run(
            static_cast<int>(full_ids.size()),
            [&](int pos) {
                int id = full_ids[static_cast<size_t>(pos)];
                int remain = s.max_iter - runs[id].iters;
                if (remain < 1) {
                    cont[static_cast<size_t>(pos)] = runs[id];
                    return;
                }
                OneRun more = run_em(data, s, runs[id].params, analyzed, remain);
                more.trace.insert(more.trace.begin(), runs[id].trace.begin(), runs[id].trace.end());
                more.regularized = more.regularized || runs[id].regularized;
                more.iters += runs[id].iters;
                // A ridge during the screen is part of the fit. The continuation
                // starts its own minimum at infinity, so without this the screen
                // eigenvalue is discarded.
                if (std::isfinite(runs[id].min_eigen)) {
                    more.min_eigen = std::isfinite(more.min_eigen)
                                         ? std::min(more.min_eigen, runs[id].min_eigen)
                                         : runs[id].min_eigen;
                }
                release_dense_posterior(more, s.K);
                cont[static_cast<size_t>(pos)] = std::move(more);
            },
            interrupt);
        continuation_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cont_t0).count();
        for (int i = 0; i < static_cast<int>(full_ids.size()); ++i)
            runs[full_ids[static_cast<size_t>(i)]] = std::move(cont[static_cast<size_t>(i)]);
    } else {
        for (int i = 0; i < nstarts; ++i) full_ids.push_back(i);
    }
    std::vector<StartVerdict> verdict(nstarts);
    const std::vector<double> gauss_var =
        layout.n_gaussian ? observed_gaussian_variance(data) : std::vector<double>{};
    const std::vector<double>* gauss_ref = layout.n_gaussian ? &gauss_var : nullptr;
    for (int id : full_ids) verdict[id] = verdict_of(data, s, runs[id], gauss_ref);
    auto loglik_of = [&](int id) {
        return std::isfinite(runs[id].loglik) ? runs[id].loglik : -kInf;
    };
    auto better_ll = [&](int a, int b) {
        if (b < 0) return true;
        double la = loglik_of(a);
        double lb = loglik_of(b);
        if (la > lb + 1e-12) return true;
        if (std::abs(la - lb) <= 1e-12 && a < b) return true;
        return false;
    };
    auto n_at = [&](double ll) {
        int n = 0;
        if (!std::isfinite(ll)) return 0;
        for (int id : full_ids) {
            if (runs[id].converged && std::isfinite(runs[id].loglik) &&
                std::abs(runs[id].loglik - ll) <= s.replicate_tol)
                ++n;
        }
        return n;
    };
    // The returned solution is the highest admissible converged start, whether
    // or not that likelihood was replicated. A lower replicated admissible
    // solution must not hide a higher unreplicated one. Replication of the
    // returned likelihood is a status (n_best), not a reason to return a
    // different start. A higher nonconverged or inadmissible start is recorded
    // below and is not the estimate.
    int best = -1;
    for (int id : full_ids) {
        if (!verdict[id].admissible) continue;
        if (better_ll(id, best)) best = id;
    }
    if (best < 0) {
        for (int id : full_ids)
            if (better_ll(id, best)) best = id;
    }
    if (best < 0) best = full_ids.empty() ? 0 : full_ids[0];
    double posterior_ms = 0;
    if (best >= 0 && best < nstarts && runs[best].dense_dropped) {
        const auto post_t0 = std::chrono::steady_clock::now();
        runs[best].fb = forward_backward(data, s, runs[best].params, analyzed, nullptr);
        runs[best].dense_dropped = false;
        posterior_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - post_t0).count();
    }
    g_phase.screen_ms = screen_ms;
    g_phase.continuation_ms = continuation_ms;
    g_phase.posterior_ms = posterior_ms;
    g_phase.screen_starts = nstarts;
    g_phase.full_starts = static_cast<int>(full_ids.size());
    int n_best = 0, n_conv = 0;
    for (int id : full_ids) {
        if (runs[id].converged && std::isfinite(runs[id].loglik)) n_conv += 1;
        if (runs[id].converged && std::isfinite(runs[id].loglik) &&
            std::abs(runs[id].loglik - runs[best].loglik) <= s.replicate_tol)
            n_best += 1;
    }
    FitResult r;
    r.engine = "cpp-em";
    r.schema_version = 2;
    r.interval_type = "none";
    r.has_draws = false;
    r.n_draws = 0;
    r.parameter_uncertainty = "none";
    r.state_uncertainty = "plugin_posterior";
    r.loglik = runs[best].loglik;
    r.converged = runs[best].converged && std::isfinite(runs[best].loglik);
    r.regularized = runs[best].regularized;
    r.n_iter = runs[best].iters;
    r.n_starts = nstarts;
    r.n_full = static_cast<int>(full_ids.size());
    r.n_converged_starts = n_conv;
    r.n_best = n_best;
    r.n_subjects = data.N;
    r.n_analyzed = static_cast<int>(analyzed.size());
    r.n_excluded = static_cast<int>(excluded.size());
    r.excluded_subjects = excluded;
    r.n_waves = data.T;
    r.n_indicators = data.J;
    r.K = s.K;
    r.seed = s.seed;
    r.n_threads = s.n_threads;
    r.tol = s.tol;
    r.replicate_tol = s.replicate_tol;
    r.data_hash = hash_panel(data);
    r.params = runs[best].params;
    r.gamma = std::move(runs[best].fb.gamma);
    r.filtered = std::move(runs[best].fb.filtered);
    r.xi = std::move(runs[best].fb.xi);
    for (int id : full_ids) {
        if (id == best) continue;
        runs[id].fb.gamma.clear();
        runs[id].fb.gamma.shrink_to_fit();
    }
    r.measurement = layout.measurement;
    r.family = layout.family;
    r.gaussian_index = layout.gaussian_index;
    r.categorical_index = layout.categorical_index;
    r.n_gaussian = layout.n_gaussian;
    r.n_categorical = layout.n_categorical;
    r.cov_model = s.cov_model;
    r.covariance_name = layout.n_gaussian ? covariance_name(s.cov_model) : "";
    r.measurement_invariant = s.measurement_invariant;
    r.means_invariant = s.means_invariant;
    r.covariance_invariant = s.covariance_invariant;
    r.transition_equal = s.transition_equal;
    r.n_tau_blocks = n_tau_blocks(data, s);
    r.n_meas_blocks = n_meas_blocks(data, s);
    r.n_mean_blocks = n_mean_blocks(data, s);
    r.n_cov_blocks = n_cov_blocks(data, s);
    r.Rmax = data.Rmax;
    r.n_parameters = count_parameters(data, s);
    r.min_eigenvalue = std::isfinite(runs[best].min_eigen) ? runs[best].min_eigen : kNaN;
    r.trace_iter.resize(runs[best].trace.size());
    r.trace_ll = runs[best].trace;
    for (int i = 0; i < static_cast<int>(r.trace_iter.size()); ++i) r.trace_iter[i] = i + 1;
    std::vector<char> is_full(nstarts, 0);
    for (int id : full_ids) is_full[id] = 1;
    for (int id = 0; id < nstarts; ++id) {
        r.start_id.push_back(id);
        r.start_phase.push_back(is_full[id] ? 1 : 0);
        r.start_loglik.push_back(runs[id].loglik);
        r.start_iters.push_back(runs[id].iters);
        r.start_converged.push_back(runs[id].converged ? 1 : 0);
        r.start_selected.push_back(id == best ? 1 : 0);
        r.start_admissible.push_back(is_full[id] && verdict[id].admissible ? 1 : 0);
    }
    std::vector<int> higher;
    for (int id : full_ids) {
        if (id == best) continue;
        if (!std::isfinite(runs[id].loglik) || !std::isfinite(runs[best].loglik)) continue;
        if (runs[id].loglik > runs[best].loglik + s.replicate_tol) higher.push_back(id);
    }
    std::sort(higher.begin(), higher.end(), [&](int a, int b) {
        if (runs[a].loglik != runs[b].loglik) return runs[a].loglik > runs[b].loglik;
        return a < b;
    });
    std::vector<char> clustered(nstarts, 0);
    for (int id : higher) {
        if (clustered[id]) continue;
        int rep = id;
        for (int other : higher) {
            if (std::abs(runs[other].loglik - runs[id].loglik) <= s.replicate_tol) {
                clustered[other] = 1;
                if (better_ll(other, rep) || other == rep) rep = other;
            }
        }
        const int n = n_at(runs[rep].loglik);
        StartVerdict v = verdict[rep];
        std::string reason = v.reason;
        std::string level = v.level;
        if (nstarts >= 2 && n < 2) {
            if (reason.empty() || reason == "ok") reason = "best_loglik_not_replicated";
            else if (reason.find("best_loglik_not_replicated") == std::string::npos)
                reason += ",best_loglik_not_replicated";
            level = "blocking";
        }
        r.rejected_start.push_back(rep);
        r.rejected_loglik.push_back(runs[rep].loglik);
        r.rejected_n.push_back(n);
        r.rejected_level.push_back(level);
        r.rejected_reason.push_back(reason);
        r.rejected_min_class_n.push_back(v.min_class_n);
        r.rejected_delta.insert(r.rejected_delta.end(), runs[rep].params.delta.begin(),
                                runs[rep].params.delta.end());
        r.rejected_means.insert(r.rejected_means.end(), runs[rep].params.means.begin(),
                                runs[rep].params.means.end());
        r.rejected_cov.insert(r.rejected_cov.end(), runs[rep].params.cov.begin(),
                              runs[rep].params.cov.end());
        r.rejected_class_eigen.insert(r.rejected_class_eigen.end(), v.class_eigen.begin(),
                                      v.class_eigen.end());
    }
    if (s.align_labels && s.K > 1) {
        auto order = canonical_order(data, s, r.params);
        bool ident = true;
        for (int i = 0; i < s.K; ++i)
            if (order[i] != i) ident = false;
        if (!ident) {
            r = relabel_fit(data, s, std::move(r), order);
        } else {
            r.label_order = order;
        }
    } else {
        r.label_order.resize(s.K);
        std::iota(r.label_order.begin(), r.label_order.end(), 0);
    }
    // recompute path summaries after alignment (relabel already permutes if applied,
    // but viterbi was not yet filled). prevalences_from fills viterbi/modal.
    prevalences_from(data, s, r, analyzed, gauss_ref);
    return r;
}

FitResult add_starts(const Panel& data, const Spec& spec, const FitResult& previous, int n_new,
                     std::uint64_t seed) {
    if (n_new < 1) throw std::invalid_argument("add_starts requires n_new >= 1");
    if (previous.data_hash != hash_panel(data)) {
        throw std::invalid_argument("add_starts data do not match the original fit");
    }
    const IndicatorLayout add_layout = indicator_layout(data);
    const std::vector<double> add_var =
        add_layout.n_gaussian ? observed_gaussian_variance(data) : std::vector<double>{};
    const std::vector<double>* add_ref = add_layout.n_gaussian ? &add_var : nullptr;
    Spec s = spec;
    s.has_start = false;
    s.n_starts = n_new;
    s.seed = mix_seed(seed, static_cast<std::uint64_t>(previous.n_starts) + 1ULL);
    s.align_labels = false;
    // The inner fit would otherwise put the deterministic initializer in
    // start 0 again and restart variants at 1. previous.n_starts is the next
    // unused variant (start 0 of the original fit consumed no variant).
    s.start_variant_offset = std::max(previous.n_starts, 1);
    FitResult neu = fit(data, s, nullptr);
    auto shift_ids = [&](FitResult& r, int offset) {
        for (int& id : r.start_id) id += offset;
        for (int& id : r.rejected_start) if (id >= 0) id += offset;
    };
    shift_ids(neu, previous.n_starts);
    const int nstarts = previous.n_starts + neu.n_starts;
    auto reps_of = [&](double ll) {
        int n = 0;
        if (!std::isfinite(ll)) return 0;
        auto scan = [&](const FitResult& f) {
            for (size_t i = 0; i < f.start_loglik.size(); ++i) {
                if (i >= f.start_phase.size() || f.start_phase[i] != 1) continue;
                if (i >= f.start_converged.size() || !f.start_converged[i]) continue;
                if (!std::isfinite(f.start_loglik[i])) continue;
                if (std::abs(f.start_loglik[i] - ll) <= spec.replicate_tol) ++n;
            }
        };
        scan(previous);
        scan(neu);
        return n;
    };
    // Highest admissible likelihood wins. Replication does not outrank it.
    auto tier_of = [&](const FitResult& f) {
        if (!(f.admissible && std::isfinite(f.loglik))) return 0;
        return 1;
    };
    const bool take_neu = [&] {
        const int tn = tier_of(neu);
        const int tp = tier_of(previous);
        if (tn != tp) return tn > tp;
        return neu.loglik > previous.loglik + 1e-12;
    }();
    FitResult out = take_neu ? neu : previous;
    const FitResult& other = take_neu ? previous : neu;
    std::vector<int> other_selected(other.start_selected.size(), 0);
    auto pad_adm = [](const FitResult& f) {
        std::vector<int> a = f.start_admissible;
        if (a.size() != f.start_id.size()) {
            a.assign(f.start_id.size(), 0);
            for (size_t i = 0; i < f.start_selected.size() && i < a.size(); ++i)
                if (f.start_selected[i]) a[i] = f.admissible ? 1 : 0;
        }
        return a;
    };
    std::vector<int> out_adm = pad_adm(out);
    std::vector<int> other_adm = pad_adm(other);
    out.start_id.insert(out.start_id.end(), other.start_id.begin(), other.start_id.end());
    out.start_phase.insert(out.start_phase.end(), other.start_phase.begin(), other.start_phase.end());
    out.start_loglik.insert(out.start_loglik.end(), other.start_loglik.begin(), other.start_loglik.end());
    out.start_iters.insert(out.start_iters.end(), other.start_iters.begin(), other.start_iters.end());
    out.start_converged.insert(out.start_converged.end(), other.start_converged.begin(),
                               other.start_converged.end());
    out.start_selected.insert(out.start_selected.end(), other_selected.begin(), other_selected.end());
    out.start_admissible = std::move(out_adm);
    out.start_admissible.insert(out.start_admissible.end(), other_adm.begin(), other_adm.end());
    // neu was fit with align_labels false. previous may already be aligned.
    // Rejected parameters are stored in out's label order.
    FitResult other_labels = align_fit(data, spec, other, out.params, nullptr);
    if (std::isfinite(other.loglik) && std::isfinite(out.loglik) &&
        other.loglik > out.loglik + spec.replicate_tol) {
        bool already = false;
        for (double ll : out.rejected_loglik)
            if (std::abs(ll - other.loglik) <= spec.replicate_tol) already = true;
        if (!already) {
            const int n = reps_of(other.loglik);
            int sid = -1;
            for (size_t i = 0; i < other.start_selected.size() && i < other.start_id.size(); ++i)
                if (other.start_selected[i]) sid = other.start_id[i];
            std::string reason = other.status_reason.empty() ? "ok" : other.status_reason;
            std::string level = other.status == STATUS_UNUSABLE ? "blocking"
                                : other.status == STATUS_WARNED ? "warning"
                                                                 : "informational";
            if (nstarts >= 2 && n < 2) {
                if (reason == "ok") reason = "best_loglik_not_replicated";
                else if (reason.find("best_loglik_not_replicated") == std::string::npos)
                    reason += ",best_loglik_not_replicated";
                level = "blocking";
            }
            out.rejected_start.push_back(sid);
            out.rejected_loglik.push_back(other.loglik);
            out.rejected_n.push_back(n);
            out.rejected_level.push_back(level);
            out.rejected_reason.push_back(reason);
            double min_n = kInf;
            if (!other_labels.expected_membership.empty() && other_labels.K > 0) {
                for (int k = 0; k < other_labels.K && k < static_cast<int>(other_labels.expected_membership.size()); ++k)
                    min_n = std::min(min_n, other_labels.expected_membership[k]);
            }
            out.rejected_min_class_n.push_back(std::isfinite(min_n) ? min_n : kNaN);
            out.rejected_delta.insert(out.rejected_delta.end(), other_labels.params.delta.begin(),
                                      other_labels.params.delta.end());
            out.rejected_means.insert(out.rejected_means.end(), other_labels.params.means.begin(),
                                      other_labels.params.means.end());
            out.rejected_cov.insert(out.rejected_cov.end(), other_labels.params.cov.begin(),
                                    other_labels.params.cov.end());
            std::vector<double> eigen(other.K, kNaN);
            if (add_layout.n_gaussian && !other_labels.params.cov.empty() && add_ref) {
                const std::vector<double>& ref = *add_ref;
                const bool shared = spec.cov_model == COV_EEI || spec.cov_model == COV_EEE;
                const int ks = shared ? 1 : other.K;
                for (int k = 0; k < ks; ++k) {
                    Eigen::MatrixXd S;
                    load_sigma(other_labels.params, spec.cov_model, other.K, add_layout.n_gaussian, 0, k, S);
                    CovarianceDiagnostic diag = diagnose_matrix(S, ref);
                    if (shared) {
                        for (int kk = 0; kk < other.K; ++kk) eigen[kk] = diag.min_eigenvalue;
                    } else {
                        eigen[k] = diag.min_eigenvalue;
                    }
                }
            }
            out.rejected_class_eigen.insert(out.rejected_class_eigen.end(), eigen.begin(), eigen.end());
        }
    }
    int n_best = 0;
    for (size_t i = 0; i < out.start_loglik.size(); ++i) {
        if (i < out.start_phase.size() && out.start_phase[i] == 1 &&
            i < out.start_converged.size() && out.start_converged[i] &&
            std::isfinite(out.start_loglik[i]) &&
            std::abs(out.start_loglik[i] - out.loglik) <= spec.replicate_tol)
            n_best += 1;
    }
    out.n_best = n_best;
    out.n_starts = previous.n_starts + neu.n_starts;
    out.n_full = previous.n_full + neu.n_full;
    out.n_converged_starts = previous.n_converged_starts + neu.n_converged_starts;
    std::vector<int> analyzed;
    for (int i = 0; i < data.N; ++i)
        if (subject_has_data(data, add_layout, i)) analyzed.push_back(i);
    // refresh status using updated replication count
    out.n_starts = previous.n_starts + n_new;
    prevalences_from(data, spec, out, analyzed, add_ref);
    if (spec.align_labels) {
        auto order = canonical_order(data, spec, out.params);
        bool ident = true;
        for (int i = 0; i < spec.K; ++i)
            if (order[i] != i) ident = false;
        if (!ident) out = relabel_fit(data, spec, std::move(out), order);
    }
    return out;
}

namespace {
void push_diff(CompareResult& c, const std::string& parameter, int state, int wave, int variable,
               int category, double a, double b) {
    DiffRow row;
    row.parameter = parameter;
    row.state = state;
    row.wave = wave;
    row.variable = variable;
    row.category = category;
    row.estimate_a = a;
    row.estimate_b = b;
    row.difference = a - b;
    c.differences.push_back(row);
}

CompareRow row_from(const FitResult& f, const std::string& role) {
    CompareRow r;
    r.role = role;
    r.loglik = f.loglik;
    r.n_parameters = f.n_parameters;
    r.aic = f.aic;
    r.bic = f.bic;
    r.converged = f.converged ? 1 : 0;
    r.n_best = f.n_best;
    r.n_starts = f.n_starts;
    r.status = f.status;
    r.ic_supported = f.ic_supported ? 1 : 0;
    return r;
}
}  // namespace

CompareResult compare_invariance(const Panel& data, const FitResult& restricted,
                                 const FitResult& unrestricted) {
    CompareResult c;
    auto refuse = [&](const std::string& why) {
        c.refused = true;
        c.reason = why;
        c.p_value_available = false;
        c.lr_statistic = kNaN;
        return c;
    };
    if (restricted.data_hash != unrestricted.data_hash || restricted.data_hash != hash_panel(data)) {
        return refuse("invariance comparison requires both fits on the same data");
    }
    if (restricted.K != unrestricted.K || restricted.measurement != unrestricted.measurement ||
        restricted.n_waves != unrestricted.n_waves || restricted.n_indicators != unrestricted.n_indicators) {
        return refuse("state count, measurement and design must match");
    }
    if (restricted.transition_equal != unrestricted.transition_equal ||
        restricted.n_tau_blocks != unrestricted.n_tau_blocks) {
        return refuse(
            "transition specification must be held fixed. Use compare_transitions for equal-transition tests.");
    }
    const IndicatorLayout layout = indicator_layout(data);
    if (layout.n_gaussian && restricted.cov_model != unrestricted.cov_model) {
        return refuse("covariance family (across states) must match; compare one restriction at a time");
    }
    if (restricted.family != unrestricted.family) {
        return refuse("indicator families must match");
    }
    const bool resp_diff = restricted.measurement_invariant != unrestricted.measurement_invariant;
    const bool means_diff = restricted.means_invariant != unrestricted.means_invariant;
    const bool cov_diff = restricted.covariance_invariant != unrestricted.covariance_invariant;
    const int n_diff = (resp_diff ? 1 : 0) + (means_diff ? 1 : 0) + (cov_diff ? 1 : 0);
    if (n_diff != 1) {
        return refuse("compare one measurement restriction at a time");
    }
    if (resp_diff && !layout.n_categorical) {
        return refuse("response invariance applies to categorical indicators, and this panel has none");
    }
    if ((means_diff || cov_diff) && !layout.n_gaussian) {
        return refuse("mean and covariance invariance apply to the Gaussian block, and this panel has none");
    }
    if (resp_diff && !(restricted.measurement_invariant && !unrestricted.measurement_invariant)) {
        return refuse("pass the response-invariant fit first and the unconstrained fit second");
    }
    if (means_diff && !(restricted.means_invariant && !unrestricted.means_invariant)) {
        return refuse("pass the means-invariant fit first");
    }
    if (cov_diff && !(restricted.covariance_invariant && !unrestricted.covariance_invariant)) {
        return refuse("pass the covariance-invariant fit first");
    }
    if (restricted.n_starts < 2 || unrestricted.n_starts < 2) {
        c.warnings.push_back("fewer than two random starts; replication was not assessed");
    }
    if (restricted.status == STATUS_UNUSABLE || unrestricted.status == STATUS_UNUSABLE) {
        c.warnings.push_back("one or both fits are marked unusable; differences are descriptive only");
    }
    Spec spec;
    spec.K = restricted.K;
    spec.cov_model = restricted.cov_model;
    spec.measurement_invariant = restricted.measurement_invariant;
    spec.means_invariant = restricted.means_invariant;
    spec.covariance_invariant = restricted.covariance_invariant;
    spec.transition_equal = restricted.transition_equal;
    auto order = match_states(data, spec, unrestricted.params, restricted.params);
    // Relabel with the unrestricted spec. The restricted spec can mark measurement
    // invariant and then mis-index a time-varying rho.
    Spec spec_u = spec;
    spec_u.measurement_invariant = unrestricted.measurement_invariant;
    spec_u.means_invariant = unrestricted.means_invariant;
    spec_u.covariance_invariant = unrestricted.covariance_invariant;
    Params aligned = relabel_params(data, spec_u, unrestricted.params, order);
    c.models.push_back(row_from(restricted, "full_invariance"));
    c.models.push_back(row_from(unrestricted, "unconstrained"));
    c.lr_statistic = 2.0 * (unrestricted.loglik - restricted.loglik);
    c.p_value_available = false;
    c.models[0].note = "no chi-square p-value in Release 1; state count is fixed, this is not a BLRT";
    const int K = restricted.K;
    if (resp_diff) {
        int nmb = unrestricted.n_meas_blocks;
        for (int t = 0; t < nmb; ++t)
            for (int j : layout.categorical_index)
                for (int k = 0; k < K; ++k)
                    for (int r = 0; r < data.ncat[j]; ++r) {
                        double a = aligned.rho[span_index(span_index(span_index(t, data.J, j), K, k), data.Rmax, r)];
                        double b = restricted.params.rho[span_index(span_index(j, K, k), data.Rmax, r)];
                        push_diff(c, "response", k + 1, t + 1, j + 1, r, a, b);
                    }
    } else if (means_diff) {
        const int Jg = layout.n_gaussian;
        int nmb = unrestricted.n_mean_blocks;
        for (int t = 0; t < nmb; ++t)
            for (int k = 0; k < K; ++k)
                for (int g = 0; g < Jg; ++g) {
                    double a = aligned.means[span_index(span_index(t, K, k), Jg, g)];
                    double b = restricted.params.means[span_index(k, Jg, g)];
                    push_diff(c, "mean", k + 1, t + 1, layout.gaussian_index[g] + 1, -1, a, b);
                }
    } else {
        const int Jg = layout.n_gaussian;
        int one = cov_block_size(restricted.cov_model, K, Jg);
        int nu = unrestricted.n_cov_blocks;
        for (int t = 0; t < nu; ++t) {
            for (int idx = 0; idx < one; ++idx) {
                push_diff(c, "covariance", 0, t + 1, idx + 1, -1, aligned.cov[span_index(t, one, idx)],
                          restricted.params.cov[idx]);
            }
        }
    }
    return c;
}

CompareResult compare_transitions(const Panel& data, const FitResult& equal_fit,
                                  const FitResult& free_fit) {
    CompareResult c;
    auto refuse = [&](const std::string& why) {
        c.refused = true;
        c.reason = why;
        c.p_value_available = false;
        c.lr_statistic = kNaN;
        return c;
    };
    if (equal_fit.data_hash != free_fit.data_hash || equal_fit.data_hash != hash_panel(data)) {
        return refuse("transition comparison requires both fits on the same data");
    }
    if (!(equal_fit.transition_equal && !free_fit.transition_equal)) {
        return refuse("pass the equal-transition fit first and the interval-specific fit second");
    }
    if (equal_fit.K != free_fit.K || equal_fit.measurement != free_fit.measurement) {
        return refuse("state count and measurement must match");
    }
    if (equal_fit.measurement_invariant != free_fit.measurement_invariant ||
        equal_fit.means_invariant != free_fit.means_invariant ||
        equal_fit.covariance_invariant != free_fit.covariance_invariant ||
        equal_fit.cov_model != free_fit.cov_model) {
        return refuse("hold the measurement specification fixed when comparing transitions");
    }
    if (equal_fit.n_starts < 2 || free_fit.n_starts < 2)
        c.warnings.push_back("fewer than two random starts; replication was not assessed");
    if (equal_fit.status == STATUS_UNUSABLE || free_fit.status == STATUS_UNUSABLE)
        c.warnings.push_back("one or both fits are marked unusable");
    Spec spec;
    spec.K = free_fit.K;
    spec.transition_equal = false;
    spec.measurement_invariant = free_fit.measurement_invariant;
    spec.means_invariant = free_fit.means_invariant;
    spec.covariance_invariant = free_fit.covariance_invariant;
    spec.cov_model = free_fit.cov_model;
    auto order = match_states(data, spec, free_fit.params, equal_fit.params);
    Params aligned = relabel_params(data, spec, free_fit.params, order);
    c.models.push_back(row_from(equal_fit, "transitions_equal"));
    c.models.push_back(row_from(free_fit, "transitions_free"));
    c.lr_statistic = 2.0 * (free_fit.loglik - equal_fit.loglik);
    c.p_value_available = false;
    const int K = equal_fit.K;
    int ntb = free_fit.n_tau_blocks;
    for (int t = 0; t < ntb; ++t)
        for (int j = 0; j < K; ++j)
            for (int k = 0; k < K; ++k) {
                double a = aligned.tau[span_index(span_index(t, K, j), K, k)];
                double b = equal_fit.params.tau[span_index((j), K, k)];
                push_diff(c, "transition", j + 1, t + 1, k + 1, -1, a, b);
            }
    return c;
}

SimResult simulate_categorical(int N, int T, int K, int J, int ncat,
                               const std::vector<double>& delta, const std::vector<double>& tau,
                               const std::vector<double>& rho, double missing_rate,
                               std::uint64_t seed, bool invariant) {
    if (N < 1 || T < 1 || K < 1 || J < 1 || ncat < 2) {
        throw std::invalid_argument("simulate_categorical received an empty design");
    }
    Panel d;
    d.N = N;
    d.T = T;
    d.J = J;
    d.measurement = MEAS_CATEGORICAL;
    d.ncat.assign(J, ncat);
    d.Rmax = ncat;
    d.family.assign(J, ncat == 2 ? FAM_BERNOULLI : FAM_NOMINAL);
    d.cat.assign(checked_product("panel", N, T, J), -1);
    PCG rng(seed);
    for (int i = 0; i < N; ++i) {
        int s = 0;
        double u = rng.uni(), c = 0;
        for (int k = 0; k < K; ++k) {
            c += delta[k];
            if (u <= c) {
                s = k;
                break;
            }
        }
        for (int t = 0; t < T; ++t) {
            int b = invariant ? 0 : t;
            for (int j = 0; j < J; ++j) {
                if (rng.uni() < missing_rate) continue;
                double uu = rng.uni(), cc = 0;
                int y = ncat - 1;
                for (int r = 0; r < ncat; ++r) {
                    cc += rho[span_index(span_index(span_index(b, J, j), K, s), ncat, r)];
                    if (uu <= cc) {
                        y = r;
                        break;
                    }
                }
                d.cat[cell(i, t, j, T, J)] = y;
            }
            if (t + 1 < T) {
                int tb = 0;
                double uu = rng.uni(), cc = 0;
                int ns = s;
                for (int k = 0; k < K; ++k) {
                    cc += tau[span_index(span_index(tb, K, s), K, k)];
                    if (uu <= cc) {
                        ns = k;
                        break;
                    }
                }
                s = ns;
            }
        }
    }
    SimResult sim;
    sim.panel = d;
    sim.truth.delta = delta;
    sim.truth.tau = tau;
    sim.truth.rho = rho;
    sim.spec.K = K;
    sim.spec.measurement_invariant = invariant;
    sim.spec.transition_equal = true;
    return sim;
}

SimResult simulate_gaussian(int N, int T, int K, int J, const std::vector<double>& delta,
                            const std::vector<double>& tau, const std::vector<double>& means,
                            const std::vector<double>& cov, int cov_model, double missing_rate,
                            std::uint64_t seed) {
    Panel d;
    d.N = N;
    d.T = T;
    d.J = J;
    d.measurement = MEAS_GAUSSIAN;
    d.family.assign(J, FAM_GAUSSIAN);
    d.gauss.assign(checked_product("panel", N, T, J), kNaN);
    PCG rng(seed);
    auto gauss = [&]() {
        double u1 = std::max(rng.uni(), 1e-12), u2 = rng.uni();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2 * kPi * u2);
    };
    Spec spec;
    spec.K = K;
    spec.cov_model = cov_model;
    Params par;
    par.means = means;
    par.cov = cov;
    for (int i = 0; i < N; ++i) {
        int s = 0;
        double u = rng.uni(), c = 0;
        for (int k = 0; k < K; ++k) {
            c += delta[k];
            if (u <= c) {
                s = k;
                break;
            }
        }
        for (int t = 0; t < T; ++t) {
            Eigen::MatrixXd Sigma;
            load_sigma(par, cov_model, K, J, 0, s, Sigma);
            Eigen::LLT<Eigen::MatrixXd> chol(Sigma);
            Eigen::VectorXd z(J);
            for (int j = 0; j < J; ++j) z(j) = gauss();
            Eigen::VectorXd y = chol.matrixL() * z;
            for (int j = 0; j < J; ++j) y(j) += means[span_index(s, J, j)];
            for (int j = 0; j < J; ++j) {
                if (rng.uni() < missing_rate) continue;
                d.gauss[cell(i, t, j, T, J)] = y(j);
            }
            if (t + 1 < T) {
                double uu = rng.uni(), cc = 0;
                int ns = s;
                for (int k = 0; k < K; ++k) {
                    cc += tau[span_index((s), K, k)];
                    if (uu <= cc) {
                        ns = k;
                        break;
                    }
                }
                s = ns;
            }
        }
    }
    SimResult sim;
    sim.panel = std::move(d);
    sim.truth.delta = delta;
    sim.truth.tau = tau;
    sim.truth.means = means;
    sim.truth.cov = cov;
    sim.spec.K = K;
    sim.spec.cov_model = cov_model;
    return sim;
}

std::string family_name(int family) {
    switch (family) {
        case FAM_GAUSSIAN: return "gaussian";
        case FAM_BERNOULLI: return "bernoulli";
        case FAM_NOMINAL: return "nominal";
        case FAM_ORDINAL: return "ordinal";
        case FAM_POISSON: return "poisson";
        case FAM_NEGBIN: return "negbin";
        default: return "unknown";
    }
}

int parse_family_name(const std::string& name) {
    std::string s = name;
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "gaussian" || s == "continuous" || s == "normal") return FAM_GAUSSIAN;
    if (s == "bernoulli" || s == "binary") return FAM_BERNOULLI;
    if (s == "nominal" || s == "multinomial" || s == "categorical") return FAM_NOMINAL;
    if (s == "ordinal" || s == "ordered" || s == "ordered_logit" || s == "ordered_probit") {
        throw std::invalid_argument(
            "Ordinal indicators (ordered logit or probit) are Release 2. "
            "Declare family 'nominal' only when the categories are unordered.");
    }
    if (s == "poisson" || s == "negbin" || s == "negative_binomial" || s == "count") {
        throw std::invalid_argument(
            "Count families (Poisson, negative binomial) are planned after the ordinal "
            "release and are not available.");
    }
    throw std::invalid_argument("unknown indicator family '" + name + "'");
}

IndicatorLayout indicator_layout(const Panel& d) {
    if (d.N < 1 || d.T < 1 || d.J < 1) {
        throw std::invalid_argument("panel must have at least one subject, wave and indicator");
    }
    if (d.J > 60) throw std::invalid_argument("Release 1 supports at most 60 indicators");
    if (!d.column_role.empty()) {
        if (static_cast<int>(d.column_role.size()) != d.J) {
            throw std::invalid_argument("column_role must have one entry per column");
        }
        for (int j = 0; j < d.J; ++j) {
            if (d.column_role[j] == ROLE_COVARIATE) {
                throw std::invalid_argument(
                    "column " + std::to_string(j + 1) +
                    " is a covariate. Indicators, covariates and distal outcomes are separate "
                    "inputs. Covariates are not estimated in Release 1, and this column was not dropped.");
            }
            if (d.column_role[j] == ROLE_DISTAL) {
                throw std::invalid_argument(
                    "column " + std::to_string(j + 1) +
                    " is a distal outcome. Distal outcomes are Release 3 (three-step ML or BCH) "
                    "and are not indicators. Modal-assignment regression is not available.");
            }
            if (d.column_role[j] != ROLE_INDICATOR) {
                throw std::invalid_argument("column " + std::to_string(j + 1) + " has an unknown role");
            }
        }
    }
    IndicatorLayout L;
    L.family.assign(d.J, FAM_GAUSSIAN);
    L.gaussian_position.assign(d.J, -1);
    if (d.family.empty()) {
        if (d.measurement == MEAS_MIXED) {
            throw std::invalid_argument(
                "a mixed panel requires an explicit family for every indicator. "
                "Separate Gaussian and categorical fits are not a mixed model.");
        }
        if (d.measurement == MEAS_GAUSSIAN) {
            std::fill(L.family.begin(), L.family.end(), FAM_GAUSSIAN);
        } else if (d.measurement == MEAS_CATEGORICAL) {
            if (static_cast<int>(d.ncat.size()) != d.J) {
                throw std::invalid_argument("ncat must have one entry per indicator");
            }
            for (int j = 0; j < d.J; ++j) {
                if (d.ncat[j] == 2) L.family[j] = FAM_BERNOULLI;
                else if (d.ncat[j] > 2) L.family[j] = FAM_NOMINAL;
                else {
                    throw std::invalid_argument("categorical indicator " + std::to_string(j + 1) +
                                                " needs at least 2 categories");
                }
            }
        } else {
            throw std::invalid_argument("measurement must be categorical, gaussian, or mixed");
        }
    } else {
        if (static_cast<int>(d.family.size()) != d.J) {
            throw std::invalid_argument("family must have one entry per indicator");
        }
        L.family = d.family;
    }
    int max_ncat = 1;
    for (int j = 0; j < d.J; ++j) {
        const int f = L.family[j];
        if (f == FAM_GAUSSIAN) {
            L.gaussian_position[j] = L.n_gaussian;
            L.gaussian_index.push_back(j);
            L.n_gaussian += 1;
        } else if (f == FAM_BERNOULLI || f == FAM_NOMINAL) {
            if (static_cast<int>(d.ncat.size()) != d.J) {
                throw std::invalid_argument("ncat must have one entry per indicator");
            }
            if (f == FAM_BERNOULLI && d.ncat[j] != 2) {
                throw std::invalid_argument("Bernoulli indicator " + std::to_string(j + 1) +
                                            " must have exactly 2 categories");
            }
            if (f == FAM_NOMINAL && d.ncat[j] < 2) {
                throw std::invalid_argument("nominal indicator " + std::to_string(j + 1) +
                                            " needs at least 2 categories");
            }
            L.categorical_index.push_back(j);
            L.n_categorical += 1;
            max_ncat = std::max(max_ncat, d.ncat[j]);
        } else if (f == FAM_ORDINAL) {
            throw std::invalid_argument(
                "Ordinal indicators (ordered logit or probit) are Release 2. "
                "Declare family 'nominal' only when the categories are unordered.");
        } else if (f == FAM_POISSON || f == FAM_NEGBIN) {
            throw std::invalid_argument(
                "Count families (Poisson, negative binomial) are planned after the ordinal "
                "release and are not available.");
        } else {
            throw std::invalid_argument("indicator " + std::to_string(j + 1) + " has an unknown family");
        }
    }
    if (L.n_gaussian && L.n_categorical) L.measurement = MEAS_MIXED;
    else if (L.n_gaussian) L.measurement = MEAS_GAUSSIAN;
    else L.measurement = MEAS_CATEGORICAL;
    if (!d.family.empty() && d.measurement != L.measurement) {
        throw std::invalid_argument(
            "measurement does not match the indicator families. "
            "Use categorical, gaussian, or mixed, and do not treat separate families as one block.");
    }
    const std::size_t ncell = checked_product("panel", d.N, d.T, d.J);
    if (L.n_categorical) {
        if (d.cat.size() != ncell) {
            throw std::invalid_argument("categorical data length does not match N*T*J");
        }
        if (d.Rmax < max_ncat) {
            throw std::invalid_argument("Rmax must be at least the largest category count");
        }
        for (int i = 0; i < d.N; ++i) {
            for (int t = 0; t < d.T; ++t) {
                for (int j : L.categorical_index) {
                    const int y = d.cat[cell(i, t, j, d.T, d.J)];
                    if (y < 0) continue;
                    if (y >= d.ncat[j]) {
                        throw std::invalid_argument(
                            "categorical code is not in 0 .. ncat-1 for indicator " + std::to_string(j + 1));
                    }
                }
            }
        }
    }
    if (L.n_gaussian) {
        if (d.gauss.size() != ncell) {
            throw std::invalid_argument("gaussian data length does not match N*T*J");
        }
    }
    return L;
}

void check_covariate_spec(const CovariateSpec& spec) {
    if (!spec.names.empty() || spec.on_initial || spec.on_transition) {
        throw std::invalid_argument(
            "Covariates are not estimated in Release 1. They are a separate input from "
            "indicators and from distal outcomes. This specification was not dropped or "
            "folded into the measurement model.");
    }
}

DistalResult estimate_distal(const DistalSpec& spec) {
    if (spec.name.empty()) throw std::invalid_argument("a distal outcome needs a name");
    const std::string method = spec.method;
    if (method == "modal" || method == "anova" || method == "modal_logistic" ||
        method == "hard") {
        throw std::invalid_argument(
            "Modal-assignment ANOVA and logistic regression are not provided. "
            "They treat the assigned state as if it were known. Release 3 estimates "
            "distal outcomes with three-step maximum likelihood or BCH and keeps the "
            "classification error in the result.");
    }
    if (spec.timing != OUTCOME_INITIAL && spec.timing != OUTCOME_FINAL &&
        spec.timing != OUTCOME_WAVE && spec.timing != OUTCOME_TRANSITION) {
        throw std::invalid_argument(
            "LTA distal outcome timing must be declared as initial, final, a specific wave, "
            "or a particular transition");
    }
    if (spec.timing == OUTCOME_WAVE && spec.wave < 1) {
        throw std::invalid_argument(
            "a wave-specific distal outcome must declare wave as a positive wave index");
    }
    if (spec.timing == OUTCOME_TRANSITION && (spec.from_state < 1 || spec.to_state < 1)) {
        throw std::invalid_argument(
            "a transition distal outcome must declare from_state and to_state");
    }
    if (spec.family == OUT_COUNT) {
        throw std::invalid_argument(
            "Count distal outcomes are planned after the Release 3 distal methods and are not estimated.");
    }
    if (spec.family != OUT_CONTINUOUS && spec.family != OUT_BINARY && spec.family != OUT_NOMINAL &&
        spec.family != OUT_ORDINAL) {
        throw std::invalid_argument("unknown distal outcome family");
    }
    throw std::invalid_argument(
        "Distal outcomes are Release 3. Supported result families then are continuous "
        "(state means and differences), binary (probabilities, differences and odds ratios), "
        "nominal (category probabilities and contrasts) and ordinal (ordered probabilities and "
        "contrasts). Estimation uses three-step maximum likelihood or BCH. Nothing was estimated, "
        "and modal assignment was not used as a substitute.");
}

SimResult simulate_mixed(int N, int T, int K, int Jg, const std::vector<int>& ncat,
                         const std::vector<double>& delta, const std::vector<double>& tau,
                         const std::vector<double>& means, const std::vector<double>& cov,
                         int cov_model, const std::vector<double>& rho, double missing_rate,
                         std::uint64_t seed) {
    const int Jc = static_cast<int>(ncat.size());
    const int J = Jg + Jc;
    if (N < 1 || T < 1 || K < 1 || Jg < 1 || Jc < 1) {
        throw std::invalid_argument(
            "simulate_mixed needs a Gaussian block and at least one categorical indicator");
    }
    int Rmax = 2;
    for (int r : ncat) Rmax = std::max(Rmax, r);
    Panel d;
    d.N = N;
    d.T = T;
    d.J = J;
    d.measurement = MEAS_MIXED;
    d.Rmax = Rmax;
    d.ncat.assign(J, 0);
    d.family.assign(J, FAM_GAUSSIAN);
    for (int j = 0; j < Jc; ++j) {
        d.ncat[Jg + j] = ncat[j];
        d.family[Jg + j] = ncat[j] == 2 ? FAM_BERNOULLI : FAM_NOMINAL;
    }
    d.gauss.assign(checked_product("panel", N, T, J), kNaN);
    d.cat.assign(checked_product("panel", N, T, J), -1);
    PCG rng(seed);
    auto gauss = [&]() {
        double u1 = std::max(rng.uni(), 1e-12), u2 = rng.uni();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2 * kPi * u2);
    };
    Params par;
    par.means = means;
    par.cov = cov;
    par.rho = rho;
    for (int i = 0; i < N; ++i) {
        int s = 0;
        double u = rng.uni(), c = 0;
        for (int k = 0; k < K; ++k) {
            c += delta[k];
            if (u <= c) {
                s = k;
                break;
            }
        }
        for (int t = 0; t < T; ++t) {
            Eigen::MatrixXd Sigma;
            load_sigma(par, cov_model, K, Jg, 0, s, Sigma);
            Eigen::LLT<Eigen::MatrixXd> chol(Sigma);
            Eigen::VectorXd z(Jg);
            for (int g = 0; g < Jg; ++g) z(g) = gauss();
            Eigen::VectorXd y = chol.matrixL() * z;
            for (int g = 0; g < Jg; ++g) y(g) += means[span_index(s, Jg, g)];
            for (int g = 0; g < Jg; ++g) {
                if (rng.uni() < missing_rate) continue;
                d.gauss[cell(i, t, g, T, J)] = y(g);
            }
            for (int j = 0; j < Jc; ++j) {
                if (rng.uni() < missing_rate) continue;
                int R = ncat[j];
                double uu = rng.uni(), cc = 0;
                int ycat = R - 1;
                for (int r = 0; r < R; ++r) {
                    cc += rho[span_index(span_index((Jg + j), K, s), Rmax, r)];
                    if (uu <= cc) {
                        ycat = r;
                        break;
                    }
                }
                d.cat[cell(i, t, Jg + j, T, J)] = ycat;
            }
            if (t + 1 < T) {
                double uu = rng.uni(), cc = 0;
                int ns = s;
                for (int k = 0; k < K; ++k) {
                    cc += tau[span_index((s), K, k)];
                    if (uu <= cc) {
                        ns = k;
                        break;
                    }
                }
                s = ns;
            }
        }
    }
    SimResult sim;
    sim.panel = std::move(d);
    sim.truth.delta = delta;
    sim.truth.tau = tau;
    sim.truth.means = means;
    sim.truth.cov = cov;
    sim.truth.rho = rho;
    sim.spec.K = K;
    sim.spec.cov_model = cov_model;
    return sim;
}

int blrt_hardware_threads() {
    unsigned n = std::thread::hardware_concurrency();
    if (n == 0) return 1;
    return static_cast<int>(n);
}

namespace {

bool blrt_usable(const FitResult& fit) {
    return fit.admissible && fit.converged && std::isfinite(fit.loglik);
}

BlrtDrawFit blrt_one_draw(const BlrtDrawJob& job, int draw, const Spec& h0, const Spec& h1,
                          std::uint64_t base, int max_retries) {
    BlrtDrawFit out;
    out.draw = draw;
    if (!job.present) {
        out.reason = "error";
        return out;
    }
    if (max_retries < 0) throw std::invalid_argument("max_retries must be at least 0");
    std::string why = "error";
    bool embed_next = false;
    Params embed_params;
    for (int retry = 0; retry <= max_retries; ++retry) {
        int scale = retry + 1;
        Spec s0 = h0;
        Spec s1 = h1;
        s0.n_threads = 1;
        s1.n_threads = 1;
        s0.has_start = false;
        s1.has_start = false;
        s0.append_supplied_start = false;
        s1.append_supplied_start = false;
        s0.n_starts = h0.n_starts * scale;
        s1.n_starts = h1.n_starts * scale;
        s0.n_full = h0.n_full * scale;
        s1.n_full = h1.n_full * scale;
        // The scheduled alternative search stays. The recovery split is one
        // more candidate, not a replacement of that search. A negative ratio
        // is not written as zero when the attempt is still below the null.
        if (embed_next) {
            s1.append_supplied_start = true;
            s1.start = embed_params;
        }
        BlrtAttemptRec attempt;
        attempt.draw = draw;
        attempt.retry = retry;
        attempt.n_starts_h0 = start_count(s0);
        attempt.n_starts_h1 = start_count(s1);
        attempt.n_starts = attempt.n_starts_h0;
        attempt.h1_embedded = s1.append_supplied_start;
        out.attempts.push_back(attempt);
        embed_next = false;
        // Same double rounding as the R seed, which is stored as numeric.
        auto seed_of = [&](double offset) {
            double value = static_cast<double>(base) + offset + 1000.0 * static_cast<double>(draw) +
                           static_cast<double>(retry);
            return static_cast<std::uint64_t>(value);
        };
        s0.seed = seed_of(0.0);
        s1.seed = seed_of(200000.0);
        FitResult fit0;
        FitResult fit1;
        bool threw = false;
        try {
            fit0 = fit(job.panel, s0, nullptr);
            fit1 = fit(job.panel, s1, nullptr);
        } catch (const std::exception&) {
            threw = true;
        }
        if (threw) {
            why = "error";
            continue;
        }
        bool ok0 = blrt_usable(fit0);
        bool ok1 = blrt_usable(fit1);
        if (!ok0 || !ok1) {
            why = (!fit0.converged || !fit1.converged) ? "not_converged" : "inadmissible";
            continue;
        }
        // H1 below H0 by more than the EM decrease allowance is not a
        // successful draw. The negative ratio is kept out of the likelihood
        // ratios; it is not replaced by zero. A later retry may start H1 from
        // a null-embedded copy of this H0 fit.
        if (blrt_h1_materially_below(fit0.loglik, fit1.loglik)) {
            why = "h1_below_h0";
            if (retry < max_retries) {
                NullSplitLayout layout;
                const ParameterBlocks blocks = parameter_blocks(job.panel, s1);
                layout.cov_model = s1.cov_model;
                layout.n_tau = blocks.n_tau;
                layout.n_meas = blocks.n_meas;
                layout.n_mean = blocks.n_mean;
                layout.n_cov = blocks.n_cov;
                layout.n_gaussian = indicator_layout(job.panel).n_gaussian;
                layout.J = job.panel.J;
                layout.Rmax = job.panel.Rmax;
                Params embedded;
                if (blrt_split_null_state(fit0.params, s0.K, s1.K, layout, embedded)) {
                    embed_next = true;
                    embed_params = std::move(embedded);
                }
            }
            continue;
        }
        out.ok = true;
        out.reason.clear();
        out.lr = 2.0 * (fit1.loglik - fit0.loglik);
        out.h0_replicated = fit0.n_best >= 2;
        // Replication needs two alternative starts. A recovery attempt keeps
        // the scheduled search, so it is assessed when that total is at least 2.
        out.h1_replication_assessed = fit1.n_starts >= 2;
        out.h1_best_fit_replicated = out.h1_replication_assessed && fit1.n_best >= 2;
        out.h1_replicated = out.h1_best_fit_replicated;
        return out;
    }
    out.ok = false;
    out.reason = why;
    return out;
}

}  // namespace

std::vector<BlrtDrawFit> blrt_fit_draws(const std::vector<BlrtDrawJob>& jobs, const Spec& h0,
                                        const Spec& h1, std::uint64_t base, int max_retries,
                                        int n_threads, void (*interrupt)(),
                                        std::vector<WorkerLoad>* worker_loads) {
    if (worker_loads) worker_loads->clear();
    if (n_threads < 1) throw std::invalid_argument("n_threads must be at least 1");
    if (h0.n_starts < 1 || h1.n_starts < 1) throw std::invalid_argument("n_starts must be at least 1");
    const int n = static_cast<int>(jobs.size());
    std::vector<BlrtDrawFit> out(n);
    int cap = blrt_hardware_threads();
    int threads = std::min(n_threads, cap);
    if (n < 1) return out;
    threads = std::min(threads, n);
    WorkerPool pool(threads);
    pool.run(
        n,
        [&](int index) { out[static_cast<size_t>(index)] = blrt_one_draw(jobs[static_cast<size_t>(index)], index + 1, h0, h1, base, max_retries); },
        interrupt);
    if (worker_loads) {
        const auto& loads = pool.loads();
        worker_loads->reserve(loads.size());
        for (const auto& slot : loads) {
            WorkerLoad row;
            row.jobs = slot.jobs;
            row.busy_seconds = slot.busy_seconds;
            worker_loads->push_back(row);
        }
    }
    return out;
}

}  // namespace lta
