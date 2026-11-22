#ifndef LTA_LTA_HPP
#define LTA_LTA_HPP

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Shared numerical core for profylr and pyfylr.
// Eigen is used with EIGEN_DONT_PARALLELIZE. Concurrency is std::thread.
// A pool runs random starts and bootstrap draws. When estep_threads > 1 and
// the caller is not already inside that pool, the E-step also runs fixed
// subject blocks on std::thread and reduces them on the calling thread.
// A start or BLRT worker forces that E-step back to one thread. Workers do
// not call back into R or Python.

namespace lta {

// The fitter's replication test: std::abs(ll - best) <= replicate_tol.
inline bool logliks_replicate(double ll, double best, double replicate_tol) {
    if (!std::isfinite(ll) || !std::isfinite(best) || !std::isfinite(replicate_tol) || replicate_tol < 0.0)
        return false;
    return std::abs(ll - best) <= replicate_tol;
}


// Bootstrap draws still use this absolute gap. An alternative this far
// below the null is not a successful draw. EM monotonicity does not use it;
// an EM step uses em_decrease_allowance.
constexpr double kEmDecreaseAbs = 1e-5;

// An EM update is rejected when the observed log-likelihood falls by more
// than this fraction of max(|previous log-likelihood|, 1). A smaller
// decrease is retained. 1e-8 matches the former absolute 1e-5 when
// |log-likelihood| is 1e3, and it scales with the sum: a large panel is not
// failed for a drop below the rounding of that sum, and a small likelihood
// is not allowed an absolute 1e-5 that is a large fraction of it. The floor
// of 1 keeps a likelihood near zero from using a zero allowance. This is the
// estimator's rule, not a convergence tolerance.
constexpr double kEmDecreaseRel = 1e-8;

// Two class weights closer than this are a tie in canonical label order.
// 1e-8 is coarser than a double ulp of a probability and finer than the
// replication tolerance, so rounding cannot swap two nearly equal
// weights while a real difference still orders them. The tie breaks on the
// first indicator's mean, or on its last category probability.
// Weights joined by a chain of pairs inside this tolerance are one component,
// even when the ends of the chain differ by more than the tolerance. The
// component is then ordered by that tie-break. A pairwise comparison of the
// distant ends is not a transitive order, so it is not used.
constexpr double kLabelDeltaTol = 1e-8;

inline double em_decrease_allowance(double loglik) {
    return kEmDecreaseRel * std::max(std::abs(loglik), 1.0);
}

// A bootstrap alternative this far below the null is not a successful draw.
// The likelihood ratio is left unrecorded; it is not clamped to zero.
inline bool blrt_h1_materially_below(double ll0, double ll1) {
    return ll1 < ll0 - kEmDecreaseAbs;
}

// Dimension products are formed in size_t. A signed multiply wraps before a
// later cast, so the cast cannot see the overflow. A product that does not
// fit is rejected before any allocation of that product.
inline std::size_t checked_product(const char* what) { return 1; }

template <typename... Rest>
inline std::size_t checked_product(const char* what, int first, Rest... rest) {
    if (first < 0) {
        throw std::invalid_argument(std::string(what) + " has a negative dimension");
    }
    const std::size_t factor = static_cast<std::size_t>(first);
    const std::size_t rest_product = checked_product(what, rest...);
    if (factor != 0 && rest_product > std::numeric_limits<std::size_t>::max() / factor) {
        throw std::invalid_argument(std::string(what) + " exceeds the supported size");
    }
    return factor * rest_product;
}

// (i * span + within) * inner + last. Valid only after checked_product has
// accepted the full dimensions that bound i, span, within, and inner.
inline std::size_t flat_index(int i, int span, int within, int inner, int last) {
    return (static_cast<std::size_t>(i) * static_cast<std::size_t>(span) + static_cast<std::size_t>(within)) *
               static_cast<std::size_t>(inner) +
           static_cast<std::size_t>(last);
}

inline std::size_t cell_index(int i, int t, int j, int T, int J) {
    return flat_index(i, T, t, J, j);
}

inline std::size_t wave_index(int i, int T, int t) {
    return static_cast<std::size_t>(i) * static_cast<std::size_t>(T) + static_cast<std::size_t>(t);
}

// block * span + within, with the product formed in size_t. The dimensions
// that bound block and span have already been accepted by checked_product
// when they size an allocation. A signed multiply would wrap before the
// index is widened.
inline std::size_t span_index(int block, int span, int within) {
    return static_cast<std::size_t>(block) * static_cast<std::size_t>(span) + static_cast<std::size_t>(within);
}

inline std::size_t span_index(std::size_t block, int span, int within) {
    return block * static_cast<std::size_t>(span) + static_cast<std::size_t>(within);
}

// Lower-triangular packed offset a*(a+1)/2 + b. The product is formed in size_t.
inline std::size_t packed_index(int a, int b) {
    return static_cast<std::size_t>(a) * static_cast<std::size_t>(a + 1) / 2u + static_cast<std::size_t>(b);
}

// ((i * intervals + t) * K + from) * K + to
inline std::size_t xi_index(int i, int intervals, int t, int K, int from, int to) {
    return flat_index(i, intervals, t, K, from) * static_cast<std::size_t>(K) + static_cast<std::size_t>(to);
}

// -inf is a zero probability. NaN and +inf are not. They fail before `out`
// is written. Every weight non-finite, or a non-positive sum, returns false
// and leaves `out` unchanged.
inline bool normalize_log_weights(const double* log_w, int n, double* out) {
    for (int i = 0; i < n; ++i) {
        const double w = log_w[i];
        if (std::isnan(w) || w == std::numeric_limits<double>::infinity()) {
            throw std::invalid_argument("log weight is NaN or +inf; only -inf is a zero probability");
        }
    }
    double m = -std::numeric_limits<double>::infinity();
    int nfin = 0;
    for (int i = 0; i < n; ++i) {
        if (log_w[i] == -std::numeric_limits<double>::infinity()) continue;
        m = std::max(m, log_w[i]);
        ++nfin;
    }
    if (nfin == 0 || !std::isfinite(m)) return false;
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        const double e = log_w[i] == -std::numeric_limits<double>::infinity() ? 0.0 : std::exp(log_w[i] - m);
        out[i] = e;
        s += e;
    }
    if (!(s > 0.0) || !std::isfinite(s)) return false;
    for (int i = 0; i < n; ++i) out[i] /= s;
    return true;
}

enum Measurement { MEAS_CATEGORICAL = 0, MEAS_GAUSSIAN = 1, MEAS_MIXED = 2 };
// Stored keeps the dense posterior for every E-step. Streaming accumulates
// categorical sufficient statistics one subject at a time and does not keep
// gamma, filtered, or xi during EM. Gaussian and mixed models have no
// streaming M-step yet.
enum EStep { ESTEP_STORED = 0, ESTEP_STREAMING = 1 };
// Per-indicator families. Ordinal is Release 2. Count families are later.
// Bernoulli and nominal are locally independent given the state. Every
// Gaussian indicator belongs to one block with the requested covariance.
enum Family {
    FAM_GAUSSIAN = 0,
    FAM_BERNOULLI = 1,
    FAM_NOMINAL = 2,
    FAM_ORDINAL = 3,
    FAM_POISSON = 4,
    FAM_NEGBIN = 5
};
enum ColumnRole { ROLE_INDICATOR = 0, ROLE_COVARIATE = 1, ROLE_DISTAL = 2 };
enum Covariance { COV_EEI = 0, COV_VVI = 1, COV_EEE = 2, COV_VVV = 3 };
enum Status { STATUS_OK = 0, STATUS_WARNED = 1, STATUS_UNUSABLE = 2 };
enum OutcomeFamily {
    OUT_CONTINUOUS = 0,
    OUT_BINARY = 1,
    OUT_NOMINAL = 2,
    OUT_ORDINAL = 3,
    OUT_COUNT = 4
};
enum OutcomeTiming {
    OUTCOME_INITIAL = 0,
    OUTCOME_FINAL = 1,
    OUTCOME_WAVE = 2,
    OUTCOME_TRANSITION = 3
};

struct Panel {
    int N = 0;
    int T = 0;
    int J = 0;
    int measurement = MEAS_CATEGORICAL;
    // Empty: infer Bernoulli (2 levels) or nominal (>2) for a categorical
    // panel, or Gaussian for a Gaussian panel. Mixed panels must set this.
    std::vector<int> family;
    // Empty: every column is an indicator. Covariate and distal columns are
    // refused; those roles are separate inputs and are not estimated here.
    std::vector<int> column_role;
    std::vector<int> ncat;     // length J; 0 or ignored for gaussian columns
    int Rmax = 1;
    std::vector<int> cat;      // N*T*J when any categorical column is present
    std::vector<double> gauss; // N*T*J when any gaussian column is present
};

// Resolved indicator layout. Gaussian columns form one block of size
// n_gaussian; categorical columns stay conditionally independent of that
// block and of each other given the state. Means and covariances are stored
// in Gaussian-block order, not in the full indicator order.
struct IndicatorLayout {
    int measurement = MEAS_CATEGORICAL;
    int n_gaussian = 0;
    int n_categorical = 0;
    std::vector<int> family;
    std::vector<int> gaussian_index;      // original 0-based indicator
    std::vector<int> categorical_index;
    std::vector<int> gaussian_position;   // length J, -1 if not gaussian
};

// Validated view of one panel. Built once from the raw arrays. The schema
// hash covers layout and the wave count, not the observed values and not N,
// so prediction can require the training schema. The data hash covers the
// values. gauss_pattern is the observed-Gaussian bit mask of each subject-wave.
struct PreparedPanel {
    IndicatorLayout layout;
    int N = 0;
    int T = 0;
    int J = 0;
    std::size_t cell_stride = 0;
    std::size_t subject_stride = 0;
    std::vector<int> analyzed;
    std::vector<int> excluded;
    std::vector<std::uint64_t> gauss_pattern;
    std::string data_hash;
    std::string schema_hash;
};

// How many parameter blocks this spec uses on a prepared panel.
struct ParameterBlocks {
    int n_tau = 0;
    int n_meas = 0;
    int n_mean = 0;
    int n_cov = 0;
};

struct Params {
    std::vector<double> delta;  // K
    std::vector<double> tau;    // n_tau_blocks * K * K, row-stochastic (from, to)
    std::vector<double> rho;    // n_meas_blocks * J * K * Rmax; gaussian columns unused
    std::vector<double> means;  // n_mean_blocks * K * n_gaussian, block order
    std::vector<double> cov;    // packed Gaussian block, layout documented in spec
};

// Block counts and covariance model for blrt_split_null_state. Counts follow
// parameter_blocks: one block when that part of the model is invariant, and
// one block per wave (or per interval, for transitions) when it is free.
struct NullSplitLayout {
    int cov_model = COV_EEI;
    int n_tau = 0;
    int n_meas = 0;
    int n_mean = 0;
    int n_cov = 0;
    int n_gaussian = 0;
    int J = 0;
    int Rmax = 0;
};

// Likelihood-preserving split of state 0. The new state copies state 0's
// emissions, means, and class covariance. The initial weight and every
// transition into state 0 are halved, and the new state's outgoing row is
// that updated row. In exact arithmetic the observed-data likelihood equals
// the null likelihood before EM. Shared covariance (EEI, EEE) is one block
// and is copied unchanged. Class covariance (VVI, VVV) gains a copy of class
// 0. Every transition, measurement, mean, and covariance block uses the same
// rule. A vector whose length does not match `layout` returns false, and the
// caller retries without this start. This does not give the new state its
// own probability mass, and it is not an embedding of the null.
inline bool blrt_split_null_state(const Params& src, int k0, int k1, const NullSplitLayout& layout, Params& dest) {
    if (k1 != k0 + 1 || k0 < 1 || static_cast<int>(src.delta.size()) != k0) return false;
    if (layout.n_tau < 0 || layout.n_meas < 0 || layout.n_mean < 0 || layout.n_cov < 0) return false;
    if (layout.n_gaussian < 0 || layout.J < 0 || layout.Rmax < 0) return false;
    auto product3 = [](int a, int b, int c, std::size_t& out) {
        if (a < 0 || b < 0 || c < 0) return false;
        std::size_t n = 1;
        const int parts[3] = {a, b, c};
        for (int v : parts) {
            const auto u = static_cast<std::size_t>(v);
            if (u != 0 && n > std::numeric_limits<std::size_t>::max() / u) return false;
            n *= u;
        }
        out = n;
        return true;
    };
    std::size_t need = 0;
    if (!product3(layout.n_tau, k0, k0, need) || src.tau.size() != need) return false;
    const bool gaussian = layout.n_gaussian > 0;
    int cov_per_block = 0;
    if (!gaussian) {
        if (!src.means.empty() || !src.cov.empty()) return false;
    } else {
        if (layout.n_mean < 1 || layout.n_cov < 1) return false;
        if (!product3(layout.n_mean, k0, layout.n_gaussian, need) || src.means.size() != need) return false;
        const int packed = layout.n_gaussian * (layout.n_gaussian + 1) / 2;
        if (layout.cov_model == COV_EEI) cov_per_block = layout.n_gaussian;
        else if (layout.cov_model == COV_EEE) cov_per_block = packed;
        else if (layout.cov_model == COV_VVI) cov_per_block = k0 * layout.n_gaussian;
        else if (layout.cov_model == COV_VVV) cov_per_block = k0 * packed;
        else return false;
        if (cov_per_block < 0) return false;
        if (!product3(layout.n_cov, cov_per_block, 1, need) || src.cov.size() != need) return false;
    }
    if (!src.rho.empty()) {
        if (layout.n_meas < 1 || layout.J < 1 || layout.Rmax < 1) return false;
        if (!product3(layout.n_meas, layout.J, k0, need)) return false;
        if (need > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(layout.Rmax)) return false;
        need *= static_cast<std::size_t>(layout.Rmax);
        if (src.rho.size() != need) return false;
    }
    dest = Params();
    dest.delta.assign(static_cast<std::size_t>(k1), 0.0);
    dest.delta[0] = src.delta[0] * 0.5;
    dest.delta[static_cast<std::size_t>(k0)] = src.delta[0] * 0.5;
    for (int k = 1; k < k0; ++k) dest.delta[static_cast<std::size_t>(k)] = src.delta[static_cast<std::size_t>(k)];
    if (layout.n_tau > 0) {
        dest.tau.assign(static_cast<std::size_t>(layout.n_tau) * static_cast<std::size_t>(k1) * static_cast<std::size_t>(k1),
                        0.0);
        for (int b = 0; b < layout.n_tau; ++b) {
            for (int from = 0; from < k0; ++from) {
                for (int to = 0; to < k0; ++to) {
                    const double value = src.tau[span_index(span_index(b, k0, from), k0, to)];
                    const double kept = to == 0 ? value * 0.5 : value;
                    dest.tau[span_index(span_index(b, k1, from), k1, to)] = kept;
                    if (to == 0) dest.tau[span_index(span_index(b, k1, from), k1, k0)] = kept;
                }
            }
            for (int to = 0; to < k1; ++to) {
                dest.tau[span_index(span_index(b, k1, k0), k1, to)] =
                    dest.tau[span_index(span_index(b, k1, 0), k1, to)];
            }
        }
    }
    if (gaussian) {
        const int Jg = layout.n_gaussian;
        dest.means.assign(static_cast<std::size_t>(layout.n_mean) * static_cast<std::size_t>(k1) * static_cast<std::size_t>(Jg),
                          0.0);
        for (int b = 0; b < layout.n_mean; ++b) {
            for (int k = 0; k < k0; ++k) {
                for (int g = 0; g < Jg; ++g) {
                    dest.means[span_index(span_index(b, k1, k), Jg, g)] =
                        src.means[span_index(span_index(b, k0, k), Jg, g)];
                }
            }
            for (int g = 0; g < Jg; ++g) {
                dest.means[span_index(span_index(b, k1, k0), Jg, g)] =
                    src.means[span_index(span_index(b, k0, 0), Jg, g)];
            }
        }
        const bool shared = layout.cov_model == COV_EEI || layout.cov_model == COV_EEE;
        if (shared) {
            dest.cov = src.cov;
        } else {
            const int per = layout.cov_model == COV_VVI ? Jg : Jg * (Jg + 1) / 2;
            dest.cov.assign(static_cast<std::size_t>(layout.n_cov) * static_cast<std::size_t>(k1) * static_cast<std::size_t>(per),
                            0.0);
            for (int b = 0; b < layout.n_cov; ++b) {
                for (int k = 0; k < k0; ++k) {
                    for (int t = 0; t < per; ++t) {
                        dest.cov[span_index(span_index(b, k1, k), per, t)] =
                            src.cov[span_index(span_index(b, k0, k), per, t)];
                    }
                }
                for (int t = 0; t < per; ++t) {
                    dest.cov[span_index(span_index(b, k1, k0), per, t)] =
                        src.cov[span_index(span_index(b, k0, 0), per, t)];
                }
            }
        }
    }
    if (!src.rho.empty()) {
        const int J = layout.J;
        const int R = layout.Rmax;
        dest.rho.assign(static_cast<std::size_t>(layout.n_meas) * static_cast<std::size_t>(J) *
                            static_cast<std::size_t>(k1) * static_cast<std::size_t>(R),
                        0.0);
        for (int b = 0; b < layout.n_meas; ++b) {
            for (int j = 0; j < J; ++j) {
                for (int k = 0; k < k0; ++k) {
                    for (int r = 0; r < R; ++r) {
                        dest.rho[span_index(span_index(span_index(b, J, j), k1, k), R, r)] =
                            src.rho[span_index(span_index(span_index(b, J, j), k0, k), R, r)];
                    }
                }
                for (int r = 0; r < R; ++r) {
                    dest.rho[span_index(span_index(span_index(b, J, j), k1, k0), R, r)] =
                        src.rho[span_index(span_index(span_index(b, J, j), k0, 0), R, r)];
                }
            }
        }
    }
    return true;
}

// Within-fit E-step workers. 1024 is below every 32-bit INT_MAX, so an
// accepted value cannot become NA when R stores it as an integer.
constexpr int kMaxEstepThreads = 1024;
constexpr int PATTERNS_NONE = 0;
constexpr int PATTERNS_COMPRESS = 1;

struct Spec {
    int K = 2;
    int cov_model = COV_VVV;
    bool measurement_invariant = true;
    bool means_invariant = true;
    bool covariance_invariant = true;
    bool transition_equal = true;
    int n_starts = 40;
    int screen_iter = 10;
    int n_full = 4;
    int max_iter = 200;
    // Relative change (ll - previous) / (|previous| + 1e-8) below `tol` stops
    // a start and marks it converged. A drop larger than
    // `em_decrease_allowance(previous)` is rejected before that test. A
    // smaller drop is kept, and it is converged when the relative change is
    // below `tol`.
    double tol = 1e-7;
    // Same solution when std::abs(ll - best) <= replicate_tol.
    // The default 0.1 is documented in docs/spec.md. A caller can set it.
    double replicate_tol = 0.1;
    int n_threads = 1;
    std::uint64_t seed = 1;
    bool align_labels = true;
    bool has_start = false;
    Params start;
    // When set without has_start, the scheduled n_starts initializers still
    // run and `start` is one more candidate. has_start still means that the
    // supplied parameters are the only start. A bootstrap retry uses this so
    // the recovery split does not replace the alternative search.
    bool append_supplied_start = false;
    // Zero keeps the historical schedule: start 0 is the deterministic
    // initializer and random variant k is start id k. add_starts sets this
    // to the previous fit's n_starts (at least 1). The new batch is then
    // entirely random, and variant = offset + local id, so pattern, partial,
    // and tail slots do not repeat the earlier fit.
    int start_variant_offset = 0;
    // "stored" is the historical E-step. "streaming" is categorical only.
    int estep = ESTEP_STORED;
    // When estep is streaming, a true value runs one posterior pass after
    // convergence. The stored E-step always returns posteriors; this flag
    // does not change that path.
    bool posteriors = true;
    // 1 keeps the historical subject loop. Values above 1, and at most
    // kMaxEstepThreads, cut the analyzed subjects into fixed blocks and
    // reduce those blocks in block order. The block size does not depend
    // on this count.
    int estep_threads = 1;
    // "none" keeps one forward-backward per analyzed subject. "compress"
    // collapses identical categorical response patterns. Appended last so a
    // default spec does not change the fields above.
    int patterns = PATTERNS_NONE;
};

struct FitResult {
    int status = STATUS_UNUSABLE;
    std::string status_reason;
    std::string engine = "cpp-em";
    int schema_version = 1;
    double loglik = 0.0;
    bool converged = false;
    bool regularized = false;
    bool ic_supported = false;
    int n_iter = 0;
    int n_starts = 0;
    int n_full = 0;
    int n_converged_starts = 0;
    // Starts at the returned likelihood. n_best >= 2 is best_fit_replicated.
    // n_starts >= 2 only means replication was assessed. Attempting several
    // starts does not establish that the best likelihood was replicated.
    int n_best = 0;
    int n_parameters = 0;
    int n_subjects = 0;
    int n_analyzed = 0;
    int n_excluded = 0;
    int n_waves = 0;
    int n_indicators = 0;
    int K = 0;
    double aic = 0.0;
    double bic = 0.0;
    double sabic = 0.0;
    double entropy_wave = 0.0;
    double entropy_sequence = 0.0;
    double entropy_wave_raw = 0.0;
    double entropy_sequence_raw = 0.0;
    double min_occupancy = 0.0;
    // Pre-ridge minimum eigenvalue of a stored covariance, in the units of
    // that matrix. It is not a status cutoff. Scale-free conclusions are the
    // three flags below and condition_number (correlation matrix).
    double min_eigenvalue = 0.0;
    bool cov_positive_definite = true;
    bool cov_ill_conditioned = false;
    bool cov_collapsed_component = false;
    double condition_number = 0.0;
    double min_relative_variance = 1.0;
    int n_numerically_empty = 0;
    int n_empty_state_waves = 0;
    int n_negligible_transition_rows = 0;
    int n_boundary = 0;
    // The returned solution is admissible when the log-likelihood is finite,
    // the start converged, the covariance was not ridged, and every stored
    // covariance is positive definite and not collapsed. Ill-conditioning,
    // a boundary probability, and a numerically empty state do not clear it.
    bool admissible = false;
    // Procedure-specific eligibility. A warning can leave the fit usable
    // while blocking one procedure. further_starts stays eligible.
    bool eligible_model_comparison = false;
    bool eligible_assignment = false;
    bool eligible_measurement = false;
    bool eligible_transitions = false;
    bool eligible_further_starts = true;
    std::string eligibility_detail;
    std::vector<double> viterbi_logprob;  // log joint probability of the path, length N
    std::uint64_t seed = 0;
    int n_threads = 1;
    double tol = 0.0;
    double replicate_tol = 0.0;
    std::string data_hash;
    std::vector<int> excluded_subjects;
    Params params;
    int n_tau_blocks = 0;
    int n_meas_blocks = 0;
    int n_mean_blocks = 0;
    int n_cov_blocks = 0;
    int Rmax = 1;
    int measurement = 0;
    std::vector<int> family;
    std::vector<int> gaussian_index;
    std::vector<int> categorical_index;
    int n_gaussian = 0;
    int n_categorical = 0;
    int cov_model = 0;
    bool measurement_invariant = true;
    bool means_invariant = true;
    bool covariance_invariant = true;
    bool transition_equal = true;
    std::string covariance_name;
    // N includes excluded subjects. Excluded rows are NaN / -1.
    std::vector<double> gamma;     // smoothed P(S_t | Y_1:T), N*T*K
    std::vector<double> filtered;  // P(S_t | Y_1:t), N*T*K
    std::vector<double> xi;        // P(S_t, S_{t+1} | Y), N*(T-1)*K*K
    std::vector<int> viterbi;      // MAP joint sequence, N*T
    std::vector<int> modal;        // smoothed marginal mode, N*T
    std::vector<int> label_order;  // new index -> old index
    std::vector<double> prevalence_model;
    std::vector<double> prevalence_posterior;
    std::vector<double> prevalence_hard;
    std::vector<double> expected_membership;
    std::vector<double> expected_transitions;
    std::vector<double> hard_transitions;
    std::vector<int> start_id;
    std::vector<int> start_phase;
    std::vector<double> start_loglik;
    std::vector<int> start_iters;
    std::vector<int> start_converged;
    std::vector<int> start_selected;
    std::vector<int> start_admissible;
    // Full starts whose log-likelihood is higher than the returned solution.
    // They are not the estimate. level is "blocking" or "warning" using the
    // same rules as a fit. n is how many full starts landed within
    // replicate_tol of that point. Parameter vectors are concatenated in
    // rejected-point order, one slot per point, in the EM label order.
    std::vector<int> rejected_start;
    std::vector<double> rejected_loglik;
    std::vector<int> rejected_n;
    std::vector<std::string> rejected_level;
    std::vector<std::string> rejected_reason;
    std::vector<double> rejected_delta;
    std::vector<double> rejected_means;
    std::vector<double> rejected_cov;
    std::vector<double> rejected_class_eigen;
    std::vector<double> rejected_min_class_n;
    std::vector<int> trace_iter;
    std::vector<double> trace_ll;
    std::vector<double> miss_indicator_n;
    std::vector<double> miss_indicator_rate;
    std::vector<double> miss_wave_n;
    std::vector<double> miss_wave_rate;
    std::vector<std::string> miss_pattern;
    std::vector<int> miss_pattern_n;
    int n_monotone = 0;
    int n_intermittent = 0;
    int n_complete_subjects = 0;
    // Parameter uncertainty is separate from state-assignment uncertainty.
    // gamma, filtered, xi, modal and viterbi are plug-in state posteriors.
    // They are not integrated over a parameter posterior or a sampling
    // distribution. Release 1 leaves draws empty. A later Bayesian backend
    // fills draws and sets interval_type to "credible". A later
    // information-matrix or bootstrap backend would set "confidence".
    bool has_draws = false;
    int n_draws = 0;
    std::vector<double> draws;
    std::string parameter_uncertainty = "none";
    std::string state_uncertainty = "plugin_posterior";
    std::string interval_type = "none";
    // Trailing so a stored fit does not change the layout of the fields above.
    // Exported only when estep is streaming. A stored fit leaves both at the
    // defaults and does not consult them.
    bool posteriors_computed = true;
    int estep = ESTEP_STORED;
    // Exported only when patterns is compress. Empty and zero on a default fit.
    int patterns = PATTERNS_NONE;
    int n_patterns = 0;
    std::vector<int> pattern_id;
};

struct BuildInfo {
    std::string eigen_version;
    std::string compiler;
    std::string flags;
    std::string cxx_standard;
    bool eigen_dont_parallelize = true;
    bool external_blas = false;
};

struct SimResult {
    Panel panel;
    Params truth;
    Spec spec;
};

BuildInfo build_info();

int count_parameters(const Panel& data, const Spec& spec);
std::string covariance_name(int cov_model);
int parse_covariance_name(const std::string& name);  // throws if unsupported
std::string family_name(int family);
int parse_family_name(const std::string& name);  // ordinal and counts throw
IndicatorLayout indicator_layout(const Panel& data);
PreparedPanel prepare_data(const Panel& data);
ParameterBlocks parameter_blocks(const Panel& data, const Spec& spec);
void require_compatible_schema(const PreparedPanel& training, const PreparedPanel& prediction);

// Covariates are a separate role from indicators. Release 1 accepts an empty
// specification and refuses a non-empty one. Nothing is silently dropped.
struct CovariateSpec {
    std::vector<std::string> names;
    bool on_initial = false;
    bool on_transition = false;
};
void check_covariate_spec(const CovariateSpec& spec);

// Distal outcomes are Release 3. The result type is declared now so a later
// three-step or BCH estimator can fill it without changing the schema.
// Modal-assignment ANOVA or logistic regression is not a method.
struct DistalSpec {
    std::string name;
    int family = OUT_CONTINUOUS;
    int timing = -1;     // must be declared: initial, final, wave, transition
    int wave = -1;       // 1-based, required when timing is a specific wave
    int from_state = -1; // 1-based, required for a transition outcome
    int to_state = -1;
    std::string method;  // empty until Release 3; "modal" is refused
};

struct DistalResult {
    bool estimated = false;
    std::string refusal;
    std::string outcome;
    std::string family;
    std::string timing;
    int wave = -1;
    int from_state = -1;
    int to_state = -1;
    std::string method;
    std::string interval_type = "none";
    std::string parameter_uncertainty = "none";
    bool classification_error_adjusted = false;
    int n_states = 0;
    int n_categories = 0;
    std::vector<double> state_mean;
    std::vector<double> mean_difference;
    std::vector<double> probability;
    std::vector<double> probability_difference;
    std::vector<double> odds_ratio;
    std::vector<double> category_probability;
    std::vector<double> contrast;
    std::vector<double> expected_count;
    std::vector<double> rate_ratio;
};

DistalResult estimate_distal(const DistalSpec& spec);

// `interrupt`, when set, is invoked on the calling thread after each wave of
// completed starts. Worker threads never call it and never touch R or Python.
// One persistent pool runs the screen and, after a full barrier, the
// continuation. `n_threads == 1` does not spawn workers.
FitResult fit(const Panel& data, const Spec& spec, void (*interrupt)() = nullptr);

// Wall time of the most recent fit() on this process. Concurrent fits
// overwrite the same record. Nested one-thread fits inside a draw pool do
// the same. The large-benchmark driver reads it after a serial fit().
struct FitPhaseTiming {
    double screen_ms = 0;
    double continuation_ms = 0;
    double posterior_ms = 0;
    int screen_starts = 0;
    int full_starts = 0;
};

FitPhaseTiming last_fit_phase_timing();

// Test seams. Not used by estimation. `fail_after < 0` disables the fault.
// A non-negative value makes the next pool constructor throw after that many
// worker threads have been started, joining any that already exist.
void debug_set_thread_creation_fail_after(int fail_after);
void debug_parallel_for(int n, int nthreads, void (*job)(int, void*), void* ctx,
                        void (*interrupt)() = nullptr);

// Test seam. Compares the likelihood-only forward with the full
// forward-backward at these parameters, including every subject value.
// True only when the 64-bit patterns match.
bool debug_likelihood_matches_full(const Panel& data, const Spec& spec, const Params& params);
double observed_loglik(const Panel& data, const Spec& spec, const Params& params);

// Fixed-parameter forward-backward. `gamma` is the smoothed state posterior,
// layout N*T*K. This does not run EM and does not align labels.
struct FixedForwardBackward {
    double loglik = 0;
    std::vector<double> gamma;
};

FixedForwardBackward fixed_forward_backward(const Panel& data, const Spec& spec, const Params& params);

// Scaled forward-backward for comparison. EM does not call this.
// Emissions are shifted by their maximum log value before exp, so a very
// small density does not underflow the whole forward vector. An observation
// that is impossible in every state (every emission non-finite, or a zero
// categorical probability on the observed category) makes the scale zero and
// the log-likelihood non-finite. `gamma` rows are divided by their sum.
struct ScaledForwardBackward {
    double loglik = 0;
    std::vector<double> gamma;
    std::vector<double> filtered;
    std::vector<double> xi;
    std::vector<double> subject;
};

ScaledForwardBackward scaled_forward_backward(const Panel& data, const Spec& spec, const Params& params);

// One EM run from an explicit start. Records the raw E-step likelihood before
// a decrease is replaced by the previous value. Not a second estimator.
struct DebugEmStep {
    int iteration = 0;
    double evaluated = 0;
    double kept = 0;
    bool decreased = false;
    bool nonfinite = false;
    bool regularized_emission = false;
    bool regularized_mstep = false;
};

struct DebugEmRun {
    std::vector<DebugEmStep> steps;
    Params params;
    double loglik = 0;
    bool converged = false;
    bool regularized = false;
};

DebugEmRun debug_em_from_params(const Panel& data, const Spec& spec, const Params& init, int max_iter);

// Log-space posterior including the transition joint. EM does not call this.
// `gamma` is the unnormalized exp(alpha+beta-loglik) the E-step stores.
struct LogPosterior {
    double loglik = 0;
    std::vector<double> gamma;
    std::vector<double> xi;
};

LogPosterior debug_log_posterior(const Panel& data, const Spec& spec, const Params& params);
// Seconds spent in the extra centered forward of the last forward_backward
// on this calling thread. Raw categorical weights that do not sum to 1 are
// accepted by debug_log_posterior and rejected by the public likelihood functions.
double debug_last_second_forward_seconds();
// Commit recorded when this library was compiled. "unset" if the build did
// not pass -DLTA_SOURCE_SHA. This is not a caller-supplied label.
const char* debug_library_source_sha();
// One observed-data log-likelihood per subject. Excluded subjects are NaN.
// The sum of the finite entries is observed_loglik.
std::vector<double> subject_loglik(const Panel& data, const Spec& spec, const Params& params);

// Continue the start id sequence so new streams do not repeat old ones.
// The added starts skip the deterministic initializer and take variants
// previous.n_starts, previous.n_starts+1, ... so they are not a second copy
// of start 0 and do not reuse the earlier partition slots.
FitResult add_starts(const Panel& data, const Spec& spec, const FitResult& previous,
                     int n_new, std::uint64_t seed);

// Test seam. One hash per start fit() would optimize, in start order.
// With start_variant_offset == 0 the first hash is the deterministic start.
std::vector<std::uint64_t> debug_start_hashes(const Panel& data, const Spec& spec);

Params align_params(const Panel& data, const Spec& spec, const Params& params,
                    std::vector<int>* order_out);
Params relabel_params(const Panel& data, const Spec& spec, const Params& params,
                      const std::vector<int>& new_to_old);
FitResult relabel_fit(const Panel& data, const Spec& spec, FitResult fit,
                      const std::vector<int>& new_to_old);

// Match `source` states onto `target` states. Returns new_to_old for source:
// index i of the aligned source is the old source state matched to target i.
// The assignment minimizes the measurement-profile distance with the Hungarian
// algorithm. Prevalence order is not this match.
std::vector<int> match_states(const Panel& data, const Spec& spec,
                              const Params& source, const Params& target);

// Distance: categorical response probabilities, squared Euclidean, plus
// Gaussian means divided by the pooled n-1 standard deviation of that
// indicator. Covariances are not in the distance. Time-varying profiles are
// averaged across blocks with equal weight. `ambiguous` is true when the
// best and second-best assignments are within 1e-4 of each other relative
// to 1 + cost.
struct Alignment {
    std::vector<int> permutation;  // new_to_old
    double cost = 0;
    double runner_up_cost = 0;
    bool ambiguous = false;
    std::string distance;
    std::string note;
};

Alignment align_states(const Panel& data, const Spec& spec, const Params& source,
                       const Params& reference);
FitResult align_fit(const Panel& data, const Spec& spec, FitResult fit,
                    const Params& reference, Alignment* info = nullptr);

// Forward-backward and Viterbi at fixed parameters. No EM.
FitResult evaluate(const Panel& data, const Spec& spec, const Params& params);

// Scale-free covariance classification. `sigma` is row-major p by p.
// `reference_scale` is one positive variance per row (the indicator's own
// spread). Empty means "use this matrix's diagonal", so a pure change of
// units is not a collapsed component.
//
// positive_definite: every eigenvalue and every variance is > 0.
// collapsed_component: a variance is <= 0, a variance is below 1e-8 of the
//   reference scale, or a correlation-matrix eigenvalue is below 1e-8.
// ill_conditioned: positive definite, not collapsed, and the correlation
//   condition number exceeds 1e6.
// level is "blocking" when the matrix is not positive definite, "warning"
// when it is collapsed or ill-conditioned, and "informational" otherwise.
// An absolute eigenvalue in (1e-10, 1e-6) does not by itself set the level.
struct CovarianceDiagnostic {
    bool positive_definite = true;
    bool ill_conditioned = false;
    bool collapsed_component = false;
    double min_eigenvalue = 0;
    double max_eigenvalue = 0;
    double min_correlation_eigenvalue = 0;
    double condition_number = 1;
    double min_relative_variance = 1;
    std::string level;
    std::string reason;
};

CovarianceDiagnostic diagnose_covariance(const std::vector<double>& sigma, int p,
                                         const std::vector<double>& reference_scale = {});

struct CompareRow {
    std::string role;
    double loglik = 0;
    int n_parameters = 0;
    double aic = 0;
    double bic = 0;
    int converged = 0;
    int n_best = 0;
    int n_starts = 0;
    int status = 0;
    int ic_supported = 0;
    std::string note;
};

struct DiffRow {
    std::string parameter;
    int state = 0;
    int wave = -1;  // -1 = invariant / shared
    int variable = 0;
    int category = -1;
    double estimate_a = 0;
    double estimate_b = 0;
    double difference = 0;
};

struct CompareResult {
    bool refused = false;
    std::string reason;
    std::vector<std::string> warnings;
    std::vector<CompareRow> models;
    std::vector<DiffRow> differences;
    // No p-value in Release 1. lr is reported only as a descriptive gap.
    double lr_statistic = 0;  // 2*(ll_free - ll_restricted); NaN if refused
    bool p_value_available = false;
};

CompareResult compare_invariance(const Panel& data, const FitResult& restricted,
                                 const FitResult& unrestricted);
CompareResult compare_transitions(const Panel& data, const FitResult& equal_fit,
                                  const FitResult& free_fit);

SimResult simulate_categorical(int N, int T, int K, int J, int ncat,
                               const std::vector<double>& delta,
                               const std::vector<double>& tau,
                               const std::vector<double>& rho,
                               double missing_rate, std::uint64_t seed,
                               bool invariant = true);

SimResult simulate_gaussian(int N, int T, int K, int J,
                            const std::vector<double>& delta,
                            const std::vector<double>& tau,
                            const std::vector<double>& means,
                            const std::vector<double>& cov, int cov_model,
                            double missing_rate, std::uint64_t seed);

// Gaussian block in the leading Jg columns, categorical indicators after.
// rho is indexed on the full J (gaussian columns unused). Conditional
// independence across the two blocks given the state.
SimResult simulate_mixed(int N, int T, int K, int Jg,
                         const std::vector<int>& ncat,
                         const std::vector<double>& delta,
                         const std::vector<double>& tau,
                         const std::vector<double>& means,
                         const std::vector<double>& cov, int cov_model,
                         const std::vector<double>& rho,
                         double missing_rate, std::uint64_t seed);

// ---- descriptives (separate translation unit; not used by EM) ----

struct DescRow {
    int variable = 0;
    int wave = -1;
    int category = -1;
    double n = 0;
    double n_missing = 0;
    double mean = 0;
    double sd = 0;
    double median = 0;
    double q25 = 0;
    double q75 = 0;
    double min = 0;
    double max = 0;
    double skew = 0;
    double kurtosis = 0;
    double proportion = 0;
    std::string kind;  // continuous or categorical
};

std::vector<DescRow> describe_indicators(const Panel& data, bool by_wave);

struct MissSummary {
    int n_subjects = 0;
    int n_excluded = 0;
    // n_complete and n_monotone are wave-pattern counts. n_complete is a
    // subject whose every wave has at least one observed indicator.
    // n_monotone is any other subject whose wave pattern changes at most
    // once, including a leading run of missing waves and a fully missing
    // subject. n_complete_indicators requires every indicator at every wave.
    // n_monotone_dropout is an observed prefix followed by a missing suffix.
    int n_complete = 0;
    int n_monotone = 0;
    int n_intermittent = 0;
    int n_complete_indicators = 0;
    int n_monotone_dropout = 0;
    std::vector<int> excluded_subjects;
    std::vector<double> indicator_n_missing;
    std::vector<double> indicator_rate;
    std::vector<double> wave_n_fully_missing;
    std::vector<double> wave_rate;
    std::vector<std::string> pattern;
    std::vector<int> pattern_n;
    std::string note;
};

MissSummary describe_missingness(const Panel& data);

struct StdResult {
    std::vector<double> values;  // same layout as panel.gauss
    std::vector<double> center;  // J, or T*J if within wave
    std::vector<double> scale;
    std::string scope;  // pooled or within_wave
    bool warned = false;
    std::string warning;
};

StdResult standardize_indicators(const Panel& data, const std::string& scope);

struct ProfileRow {
    int variable = 0;
    int state = 0;
    int wave = -1;
    int category = -1;
    std::string source;  // model or posterior_weighted
    std::string statistic;  // mean, sd, proportion
    double estimate = 0;
    double weight = 0;
    double reference_value = 0;
    double reference_sd = 0;
    double difference = 0;
    double standardized_difference = 0;
    std::string reference_sd_source;
    std::string weighting;
};

std::vector<ProfileRow> describe_profiles(const Panel& data, const Spec& spec,
                                          const FitResult& fit,
                                          const std::vector<double>& reference_sd);

// Pairwise Pearson correlations of the Gaussian indicators. Each pair uses
// the person-waves where both entries are observed. The covariance divides
// by n - 1. The correlation does not.
struct CorrRow {
    int variable_a = 0;
    int variable_b = 0;
    int wave = -1;
    int n = 0;
    double correlation = 0;
    double covariance = 0;
    std::string method;
};

std::vector<CorrRow> correlate_indicators(const Panel& data);

// One-state Gaussian fit. VVV is the saturated multivariate normal. EEI is
// the diagonal constraint. Means and covariances are maximum likelihood
// (the covariance divides by n). This is the Release 1 check against a
// saturated lavaan model with missing = "ml".
struct FimlMoments {
    std::string covariance;
    int n = 0;
    int p = 0;
    double loglik = 0;
    int n_parameters = 0;
    bool converged = false;
    std::vector<double> mean;
    std::vector<double> cov;  // p by p, row major
    std::vector<double> cor;
    std::string note;
};

FimlMoments fiml_moments(const Panel& data, const std::string& covariance);

// One bootstrap draw. The panel is already simulated. `present` is false when
// simulation failed and no fit is attempted. Draw numbers elsewhere are
// 1-based and equal to the index here plus one.
struct BlrtDrawJob {
    bool present = false;
    Panel panel;
};

struct BlrtAttemptRec {
    int draw = 0;
    int retry = 0;
    // Actual null start count. Kept under this name so existing readers of
    // the attempt row still see the null. n_starts_h1 is the alternative.
    int n_starts = 0;
    int n_starts_h0 = 0;
    int n_starts_h1 = 0;
    // True when this attempt added the likelihood-preserving split of the
    // null as one extra alternative candidate. The scheduled alternative
    // search still runs.
    bool h1_embedded = false;
};

// Result of fitting one draw, including retries on that same panel.
// `lr` is 2 (log L_1 - log L_0) of the returned admissible solutions.
struct BlrtDrawFit {
    int draw = 0;
    bool ok = false;
    std::string reason;
    double lr = 0.0;
    // Best-fit replication: at least two converged starts share the returned
    // likelihood. This is not the count of starts that were attempted.
    bool h0_replicated = false;
    bool h1_replicated = false;
    // Same value as h1_replicated. The name says which event was replicated.
    bool h1_best_fit_replicated = false;
    // True when the alternative ran at least two starts, so a replicate could
    // have been seen. Several attempted starts do not establish replication.
    // h1_best_fit_replicated stays false when this is false, and that is not
    // a failed replicate.
    bool h1_replication_assessed = true;
    std::vector<BlrtAttemptRec> attempts;
};

// Number of hardware threads, at least 1. This is the cap for draw workers.
int blrt_hardware_threads();

// One worker's share of a draw-pool call. `jobs` is the number of draws that
// worker finished. `busy_seconds` is the time it spent inside those draws.
// The core does not print this. R and Python print it when a profile is asked
// for, so the compiled library does not reference std::cout.
struct WorkerLoad {
    int jobs = 0;
    double busy_seconds = 0;
};

// Fit every supplied draw. Seeds are `base + 1000 * draw + retry` and
// `base + 200000 + 1000 * draw + retry`. `draw` is 1-based. An empty
// `draw_ids` uses the position in `jobs` plus one, which is the historical
// call with every draw in one vector. A non-empty `draw_ids` is the stable
// global id of each job, so a batch keeps the same seeds as that full vector.
// A draw does not depend on which thread runs it, how many threads were
// asked for, or which other draws are in the vector. Each fit uses one
// thread. `n_threads` is the number of draws in flight. Values above the
// hardware cap are clamped. `n_threads == 1` does not spawn threads. One
// persistent pool pulls draw indices from an atomic counter. `interrupt`,
// when set, runs on the calling thread between waves of completed draws and
// must not be called from a worker. When `worker_loads` is non-null it is
// replaced with one entry per worker.
std::vector<BlrtDrawFit> blrt_fit_draws(const std::vector<BlrtDrawJob>& jobs, const Spec& h0,
                                        const Spec& h1, std::uint64_t base, int max_retries,
                                        int n_threads, void (*interrupt)() = nullptr,
                                        std::vector<WorkerLoad>* worker_loads = nullptr,
                                        const std::vector<int>& draw_ids = {});

}  // namespace lta

#endif
