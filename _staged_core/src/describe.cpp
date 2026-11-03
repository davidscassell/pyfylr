#ifndef EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_PARALLELIZE
#endif

#include "lta/lta.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>

namespace lta {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

std::size_t cell(int i, int t, int j, int T, int J) { return cell_index(i, t, j, T, J); }

bool missing_at(const Panel& d, const IndicatorLayout& L, int i, int t, int j) {
    if (L.family[j] == FAM_GAUSSIAN) {
        return !std::isfinite(d.gauss[cell(i, t, j, d.T, d.J)]);
    }
    return d.cat[cell(i, t, j, d.T, d.J)] < 0;
}

double quantile_sorted(const std::vector<double>& x, double p) {
    if (x.empty()) return kNaN;
    double idx = p * (static_cast<double>(x.size()) - 1.0);
    int i0 = static_cast<int>(idx);
    double f = idx - i0;
    if (i0 + 1 >= static_cast<int>(x.size())) return x.back();
    return (1.0 - f) * x[i0] + f * x[i0 + 1];
}

DescRow continuous_row(const std::vector<double>& x, int variable, int wave) {
    DescRow row;
    row.variable = variable;
    row.wave = wave;
    row.category = -1;
    row.kind = "continuous";
    row.n = static_cast<double>(x.size());
    if (x.empty()) {
        row.mean = row.sd = row.median = row.q25 = row.q75 = row.min = row.max = row.skew =
            row.kurtosis = kNaN;
        return row;
    }
    double s = std::accumulate(x.begin(), x.end(), 0.0);
    row.mean = s / row.n;
    double m2 = 0, m3 = 0, m4 = 0;
    for (double v : x) {
        double d = v - row.mean;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
    }
    m2 /= row.n;
    m3 /= row.n;
    m4 /= row.n;
    row.sd = (x.size() > 1) ? std::sqrt(m2 * row.n / (row.n - 1.0)) : kNaN;
    row.skew = (m2 > 0 && x.size() >= 3) ? m3 / std::pow(m2, 1.5) : kNaN;
    row.kurtosis = (m2 > 0 && x.size() >= 4) ? m4 / (m2 * m2) - 3.0 : kNaN;
    std::vector<double> xs = x;
    std::sort(xs.begin(), xs.end());
    row.min = xs.front();
    row.max = xs.back();
    row.median = quantile_sorted(xs, 0.5);
    row.q25 = quantile_sorted(xs, 0.25);
    row.q75 = quantile_sorted(xs, 0.75);
    return row;
}

}  // namespace

std::vector<DescRow> describe_indicators(const Panel& data, bool by_wave) {
    const IndicatorLayout L = indicator_layout(data);
    std::vector<DescRow> rows;
    auto collect_gauss = [&](int j, int wave) {
        std::vector<double> x;
        int nmiss = 0;
        for (int i = 0; i < data.N; ++i) {
            int t0 = wave < 0 ? 0 : wave;
            int t1 = wave < 0 ? data.T : wave + 1;
            for (int t = t0; t < t1; ++t) {
                if (missing_at(data, L, i, t, j)) nmiss += 1;
                else x.push_back(data.gauss[cell(i, t, j, data.T, data.J)]);
            }
        }
        DescRow row = continuous_row(x, j, wave);
        row.n_missing = nmiss;
        rows.push_back(row);
    };
    auto collect_cat = [&](int j, int wave) {
        int R = data.ncat[j];
        std::vector<int> cnt(R, 0);
        int n = 0, nmiss = 0;
        for (int i = 0; i < data.N; ++i) {
            int t0 = wave < 0 ? 0 : wave;
            int t1 = wave < 0 ? data.T : wave + 1;
            for (int t = t0; t < t1; ++t) {
                if (missing_at(data, L, i, t, j)) {
                    nmiss += 1;
                    continue;
                }
                int y = data.cat[cell(i, t, j, data.T, data.J)];
                if (y >= 0 && y < R) {
                    cnt[y] += 1;
                    n += 1;
                }
            }
        }
        for (int r = 0; r < R; ++r) {
            DescRow row;
            row.variable = j;
            row.wave = wave;
            row.category = r;
            row.kind = L.family[j] == FAM_BERNOULLI ? "bernoulli" : "nominal";
            row.n = n;
            row.n_missing = nmiss;
            row.proportion = n ? static_cast<double>(cnt[r]) / n : kNaN;
            row.mean = row.sd = row.median = row.q25 = row.q75 = row.min = row.max = row.skew =
                row.kurtosis = kNaN;
            rows.push_back(row);
        }
    };
    for (int j = 0; j < data.J; ++j) {
        if (!by_wave) {
            if (L.family[j] == FAM_GAUSSIAN) collect_gauss(j, -1);
            else collect_cat(j, -1);
        } else {
            for (int t = 0; t < data.T; ++t) {
                if (L.family[j] == FAM_GAUSSIAN) collect_gauss(j, t);
                else collect_cat(j, t);
            }
        }
    }
    return rows;
}

MissSummary describe_missingness(const Panel& data) {
    const IndicatorLayout L = indicator_layout(data);
    MissSummary m;
    m.n_subjects = data.N;
    m.indicator_n_missing.assign(data.J, 0);
    m.indicator_rate.assign(data.J, 0);
    m.wave_n_fully_missing.assign(data.T, 0);
    m.wave_rate.assign(data.T, 0);
    std::map<std::string, int> patterns;
    for (int i = 0; i < data.N; ++i) {
        bool any = false;
        std::string pat;
        for (int t = 0; t < data.T; ++t) {
            bool wave_miss = true;
            for (int j = 0; j < data.J; ++j) {
                bool miss = missing_at(data, L, i, t, j);
                if (miss) m.indicator_n_missing[j] += 1;
                else {
                    wave_miss = false;
                    any = true;
                }
            }
            pat.push_back(wave_miss ? 'M' : 'O');
            if (wave_miss) m.wave_n_fully_missing[t] += 1;
        }
        if (!any) m.excluded_subjects.push_back(i);
        int changes = 0;
        for (int t = 1; t < data.T; ++t)
            if (pat[t] != pat[t - 1]) ++changes;
        // Wave-pattern counts. A wave is observed when at least one indicator
        // in that wave is observed. Leading-missing waves stay in n_monotone.
        bool wave_complete = pat.find('M') == std::string::npos;
        if (wave_complete) m.n_complete += 1;
        else if (changes <= 1) m.n_monotone += 1;
        else m.n_intermittent += 1;
        bool every_indicator = true;
        for (int t = 0; t < data.T && every_indicator; ++t)
            for (int j = 0; j < data.J; ++j)
                if (missing_at(data, L, i, t, j)) every_indicator = false;
        if (every_indicator) m.n_complete_indicators += 1;
        // Monotone dropout is an observed prefix and a missing suffix.
        const auto first_missing = pat.find('M');
        if (!pat.empty() && pat[0] == 'O' && first_missing != std::string::npos &&
            pat.find('O', first_missing) == std::string::npos)
            m.n_monotone_dropout += 1;
        patterns[pat] += 1;
    }
    m.n_excluded = static_cast<int>(m.excluded_subjects.size());
    double cells = static_cast<double>(data.N) * data.T;
    for (int j = 0; j < data.J; ++j)
        m.indicator_rate[j] = cells ? m.indicator_n_missing[j] / cells : 0;
    for (int t = 0; t < data.T; ++t)
        m.wave_rate[t] = data.N ? m.wave_n_fully_missing[t] / static_cast<double>(data.N) : 0;
    for (const auto& kv : patterns) {
        m.pattern.push_back(kv.first);
        m.pattern_n.push_back(kv.second);
    }
    std::ostringstream note;
    note << "n_complete and n_monotone are wave-pattern counts. n_complete counts "
            "subjects whose every wave has at least one observed indicator. "
            "n_monotone counts the other subjects whose wave pattern changes at most "
            "once, including a leading run of missing waves. "
            "n_complete_indicators counts subjects with every indicator observed. "
            "n_monotone_dropout counts an observed prefix followed by a missing suffix. "
            "A leading-missing pattern is not monotone dropout. ";
    note << "A fully missing wave stays in the sequence and contributes two transition "
            "steps when it sits between observed waves. ";
    if (L.measurement == MEAS_MIXED) {
        note << "Gaussian and categorical indicators are conditionally independent given "
                "the state; missingness is recorded per indicator.";
    } else {
        note << "Partial indicator vectors are retained. Only subjects with no observed "
                "indicators are excluded, and that exclusion is reported.";
    }
    m.note = note.str();
    return m;
}

StdResult standardize_indicators(const Panel& data, const std::string& scope) {
    const IndicatorLayout L = indicator_layout(data);
    StdResult out;
    out.values = data.gauss;
    out.scope = scope;
    if (scope != "pooled" && scope != "within_wave") {
        throw std::invalid_argument("standardize scope must be 'pooled' or 'within_wave'");
    }
    if (!L.n_gaussian) {
        out.warned = true;
        out.warning = "no Gaussian indicators to standardize; categorical indicators were left unchanged";
        return out;
    }
    const int blocks = scope == "pooled" ? 1 : data.T;
    out.center.assign(checked_product("flat", blocks, data.J), kNaN);
    out.scale.assign(checked_product("flat", blocks, data.J), kNaN);
    for (int b = 0; b < blocks; ++b) {
        for (int j : L.gaussian_index) {
            std::vector<double> x;
            for (int i = 0; i < data.N; ++i) {
                int t0 = scope == "pooled" ? 0 : b;
                int t1 = scope == "pooled" ? data.T : b + 1;
                for (int t = t0; t < t1; ++t) {
                    double y = data.gauss[cell(i, t, j, data.T, data.J)];
                    if (std::isfinite(y)) x.push_back(y);
                }
            }
            double mean = 0;
            for (double v : x) mean += v;
            mean = x.empty() ? 0 : mean / static_cast<double>(x.size());
            double ss = 0;
            for (double v : x) ss += (v - mean) * (v - mean);
            double sd = (x.size() > 1) ? std::sqrt(ss / (static_cast<double>(x.size()) - 1.0)) : 0;
            out.center[span_index(b, data.J, j)] = mean;
            out.scale[span_index(b, data.J, j)] = sd;
            if (!(sd > 0)) {
                out.warned = true;
                out.warning = "at least one Gaussian indicator has zero or undefined scale; those values were left unscaled";
                continue;
            }
            for (int i = 0; i < data.N; ++i) {
                int t0 = scope == "pooled" ? 0 : b;
                int t1 = scope == "pooled" ? data.T : b + 1;
                for (int t = t0; t < t1; ++t) {
                    std::size_t c = cell(i, t, j, data.T, data.J);
                    if (std::isfinite(out.values[c])) out.values[c] = (out.values[c] - mean) / sd;
                }
            }
        }
    }
    return out;
}

std::vector<ProfileRow> describe_profiles(const Panel& data, const Spec& spec, const FitResult& fit,
                                          const std::vector<double>& reference_sd) {
    const IndicatorLayout L = indicator_layout(data);
    if (fit.K != spec.K) throw std::invalid_argument("fit and spec disagree on the number of states");
    std::vector<double> ref = reference_sd;
    std::string ref_source = "declared";
    if (ref.empty()) {
        ref.assign(data.J, kNaN);
        ref_source = "pooled_sample";
        for (int j : L.gaussian_index) {
            std::vector<double> x;
            for (int i = 0; i < data.N; ++i)
                for (int t = 0; t < data.T; ++t) {
                    double y = data.gauss[cell(i, t, j, data.T, data.J)];
                    if (std::isfinite(y)) x.push_back(y);
                }
            if (x.size() < 2) continue;
            double mean = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
            double ss = 0;
            for (double v : x) ss += (v - mean) * (v - mean);
            ref[j] = std::sqrt(ss / (static_cast<double>(x.size()) - 1.0));
        }
    } else if (static_cast<int>(ref.size()) != data.J) {
        throw std::invalid_argument("reference_sd must have one entry per indicator, or be empty");
    }
    std::vector<ProfileRow> rows;
    auto push = [&](ProfileRow row) { rows.push_back(std::move(row)); };
    if (L.n_categorical && !fit.params.rho.empty()) {
        int nmb = std::max(1, static_cast<int>(fit.params.rho.size() / checked_product("flat", data.J, spec.K, data.Rmax)));
        for (int b = 0; b < nmb; ++b)
            for (int j : L.categorical_index)
                for (int k = 0; k < spec.K; ++k)
                    for (int r = 0; r < data.ncat[j]; ++r) {
                        ProfileRow row;
                        row.variable = j;
                        row.state = k;
                        row.wave = spec.measurement_invariant ? -1 : b;
                        row.category = r;
                        row.source = "model";
                        row.statistic = "proportion";
                        row.estimate = fit.params.rho[span_index(span_index(span_index(b, data.J, j), spec.K, k), data.Rmax, r)];
                        row.reference_sd_source = "not_applicable";
                        row.weighting = "model";
                        push(row);
                    }
    }
    if (L.n_gaussian && !fit.params.means.empty()) {
        const int Jg = L.n_gaussian;
        int nmb = std::max(1, static_cast<int>(fit.params.means.size() / checked_product("flat", spec.K, Jg)));
        for (int b = 0; b < nmb; ++b)
            for (int k = 0; k < spec.K; ++k)
                for (int g = 0; g < Jg; ++g) {
                    int j = L.gaussian_index[g];
                    ProfileRow row;
                    row.variable = j;
                    row.state = k;
                    row.wave = spec.means_invariant ? -1 : b;
                    row.category = -1;
                    row.source = "model";
                    row.statistic = "mean";
                    row.estimate = fit.params.means[span_index(span_index(b, spec.K, k), Jg, g)];
                    double grand = 0;
                    int ngrand = 0;
                    for (int i = 0; i < data.N; ++i)
                        for (int t = 0; t < data.T; ++t) {
                            double y = data.gauss[cell(i, t, j, data.T, data.J)];
                            if (!std::isfinite(y)) continue;
                            grand += y;
                            ngrand += 1;
                        }
                    row.reference_value = ngrand ? grand / ngrand : kNaN;
                    row.reference_sd = j < static_cast<int>(ref.size()) ? ref[j] : kNaN;
                    row.reference_sd_source = ref_source;
                    row.difference = row.estimate - row.reference_value;
                    row.standardized_difference =
                        (row.reference_sd > 0) ? row.difference / row.reference_sd : kNaN;
                    row.weighting = "model";
                    push(row);
                }
    }
    if (fit.gamma.empty()) return rows;
    for (int j = 0; j < data.J; ++j) {
        for (int k = 0; k < spec.K; ++k) {
            if (L.family[j] == FAM_GAUSSIAN) {
                double w = 0, sy = 0;
                for (int i = 0; i < data.N; ++i)
                    for (int t = 0; t < data.T; ++t) {
                        double y = data.gauss[cell(i, t, j, data.T, data.J)];
                        double g = fit.gamma[flat_index(i, data.T, t, spec.K, k)];
                        if (!std::isfinite(y) || !std::isfinite(g)) continue;
                        w += g;
                        sy += g * y;
                    }
                ProfileRow row;
                row.variable = j;
                row.state = k;
                row.wave = -1;
                row.category = -1;
                row.source = "posterior_weighted";
                row.statistic = "mean";
                row.weight = w;
                row.estimate = w > 0 ? sy / w : kNaN;
                row.reference_sd = j < static_cast<int>(ref.size()) ? ref[j] : kNaN;
                row.reference_sd_source = ref_source;
                row.weighting = "posterior_probability";
                double grand = 0;
                int ngrand = 0;
                for (int i = 0; i < data.N; ++i)
                    for (int t = 0; t < data.T; ++t) {
                        double y = data.gauss[cell(i, t, j, data.T, data.J)];
                        if (!std::isfinite(y)) continue;
                        grand += y;
                        ngrand += 1;
                    }
                row.reference_value = ngrand ? grand / ngrand : kNaN;
                row.difference = row.estimate - row.reference_value;
                row.standardized_difference =
                    (row.reference_sd > 0 && std::isfinite(row.estimate)) ? row.difference / row.reference_sd
                                                                         : kNaN;
                push(row);
                // Centered second pass. E[y^2] - E[y]^2 cancels for a large
                // location and was clamped at zero. Sum of g (y - mean)^2
                // does not depend on adding a constant to y.
                double m2 = 0;
                if (std::isfinite(row.estimate)) {
                    for (int i = 0; i < data.N; ++i)
                        for (int t = 0; t < data.T; ++t) {
                            double y = data.gauss[cell(i, t, j, data.T, data.J)];
                            double g = fit.gamma[flat_index(i, data.T, t, spec.K, k)];
                            if (!std::isfinite(y) || !std::isfinite(g)) continue;
                            double d = y - row.estimate;
                            m2 += g * d * d;
                        }
                }
                ProfileRow sd = row;
                sd.statistic = "sd";
                sd.estimate = (w > 1.0 && std::isfinite(row.estimate)) ? std::sqrt(m2 / w) : kNaN;
                sd.difference = kNaN;
                sd.standardized_difference = kNaN;
                push(sd);
            } else {
                int R = data.ncat[j];
                std::vector<double> num(R, 0.0);
                double den = 0;
                for (int i = 0; i < data.N; ++i)
                    for (int t = 0; t < data.T; ++t) {
                        int y = data.cat[cell(i, t, j, data.T, data.J)];
                        if (y < 0 || y >= R) continue;
                        double g = fit.gamma[flat_index(i, data.T, t, spec.K, k)];
                        num[y] += g;
                        den += g;
                    }
                for (int r = 0; r < R; ++r) {
                    ProfileRow row;
                    row.variable = j;
                    row.state = k;
                    row.wave = -1;
                    row.category = r;
                    row.source = "posterior_weighted";
                    row.statistic = "proportion";
                    row.weight = den;
                    row.estimate = den > 0 ? num[r] / den : kNaN;
                    row.reference_sd_source = "not_applicable";
                    row.weighting = "posterior_probability";
                    push(row);
                }
            }
        }
    }
    return rows;
}

std::vector<CorrRow> correlate_indicators(const Panel& data) {
    const IndicatorLayout L = indicator_layout(data);
    if (!L.n_gaussian) {
        throw std::invalid_argument("correlate_indicators needs at least one Gaussian indicator");
    }
    if (L.n_categorical) {
        throw std::invalid_argument(
            "correlate_indicators is the Pearson correlation of the Gaussian indicators. "
            "Categorical indicators are not scored as numeric.");
    }
    std::vector<CorrRow> rows;
    const int p = static_cast<int>(L.gaussian_index.size());
    struct Uni {
        double ss = 0;
        int n = 0;
    };
    std::vector<Uni> univariate(p);
    for (int a = 0; a < p; ++a) {
        std::vector<double> x;
        for (int i = 0; i < data.N; ++i) {
            for (int t = 0; t < data.T; ++t) {
                double value = data.gauss[cell(i, t, L.gaussian_index[a], data.T, data.J)];
                if (std::isfinite(value)) x.push_back(value);
            }
        }
        univariate[a].n = static_cast<int>(x.size());
        if (x.size() > 1) {
            double mean = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
            for (double value : x) univariate[a].ss += (value - mean) * (value - mean);
        }
    }
    for (int a = 0; a < p; ++a) {
        for (int b = a; b < p; ++b) {
            std::vector<double> x, y;
            x.reserve(data.N * data.T);
            y.reserve(data.N * data.T);
            for (int i = 0; i < data.N; ++i) {
                for (int t = 0; t < data.T; ++t) {
                    double xa = data.gauss[cell(i, t, L.gaussian_index[a], data.T, data.J)];
                    double xb = data.gauss[cell(i, t, L.gaussian_index[b], data.T, data.J)];
                    if (!std::isfinite(xa) || !std::isfinite(xb)) continue;
                    x.push_back(xa);
                    y.push_back(xb);
                }
            }
            CorrRow row;
            row.variable_a = L.gaussian_index[a];
            row.variable_b = L.gaussian_index[b];
            row.wave = -1;
            row.n = static_cast<int>(x.size());
            row.method = "pairwise";
            if (row.n < 2) {
                row.correlation = kNaN;
                row.covariance = kNaN;
            } else {
                double mx = std::accumulate(x.begin(), x.end(), 0.0) / row.n;
                double my = std::accumulate(y.begin(), y.end(), 0.0) / row.n;
                double sxx = 0, syy = 0, sxy = 0;
                for (int i = 0; i < row.n; ++i) {
                    double dx = x[i] - mx;
                    double dy = y[i] - my;
                    sxx += dx * dx;
                    syy += dy * dy;
                    sxy += dx * dy;
                }
                row.covariance = sxy / (row.n - 1.0);
                // lavCor(missing = "pairwise") divides the pairwise covariance by
                // the available-case standard deviations, not the pair's own.
                double va = univariate[a].n > 1 ? univariate[a].ss / (univariate[a].n - 1.0) : 0.0;
                double vb = univariate[b].n > 1 ? univariate[b].ss / (univariate[b].n - 1.0) : 0.0;
                row.correlation = (va > 0.0 && vb > 0.0) ? row.covariance / std::sqrt(va * vb) : kNaN;
            }
            rows.push_back(row);
        }
    }
    return rows;
}

FimlMoments fiml_moments(const Panel& data, const std::string& covariance) {
    const IndicatorLayout L = indicator_layout(data);
    if (!L.n_gaussian || L.n_categorical) {
        throw std::invalid_argument(
            "fiml_moments is the one-state Gaussian model, the saturated multivariate "
            "normal when the covariance is unrestricted. Categorical indicators are not included.");
    }
    const int cov_model = parse_covariance_name(covariance);
    if (cov_model != COV_VVV && cov_model != COV_EEI) {
        throw std::invalid_argument(
            "fiml_moments accepts VVV (unrestricted covariance) or EEI (diagonal covariance).");
    }
    Spec spec;
    spec.K = 1;
    spec.cov_model = cov_model;
    spec.n_starts = 1;
    spec.n_full = 1;
    spec.screen_iter = 0;
    spec.max_iter = 500;
    spec.tol = 1e-12;
    spec.seed = 1;
    spec.align_labels = true;
    const FitResult fitted = fit(data, spec, nullptr);
    FimlMoments moments;
    moments.covariance = covariance_name(cov_model);
    moments.n = fitted.n_analyzed;
    moments.p = L.n_gaussian;
    moments.loglik = fitted.loglik;
    moments.n_parameters = fitted.n_parameters;
    moments.converged = fitted.converged;
    moments.mean = fitted.params.means;
    moments.cov.assign(checked_product("flat", moments.p, moments.p), 0.0);
    moments.cor.assign(checked_product("flat", moments.p, moments.p), 0.0);
    if (cov_model == COV_EEI) {
        for (int j = 0; j < moments.p; ++j) moments.cov[span_index(j, moments.p, j)] = fitted.params.cov[j];
    } else {
        for (int a = 0; a < moments.p; ++a) {
            for (int b = 0; b <= a; ++b) {
                double value = fitted.params.cov[packed_index(a, b)];
                moments.cov[span_index(a, moments.p, b)] = value;
                moments.cov[span_index(b, moments.p, a)] = value;
            }
        }
    }
    for (int a = 0; a < moments.p; ++a) {
        for (int b = 0; b < moments.p; ++b) {
            double da = moments.cov[span_index(a, moments.p, a)];
            double db = moments.cov[span_index(b, moments.p, b)];
            moments.cor[span_index(a, moments.p, b)] =
                (da > 0.0 && db > 0.0) ? moments.cov[span_index(a, moments.p, b)] / std::sqrt(da * db) : kNaN;
        }
    }
    moments.note = cov_model == COV_VVV
                       ? "One-state VVV is the saturated multivariate normal. "
                         "The covariance divides by n, the same scale as lavaan with missing = 'ml'."
                       : "One-state EEI is a diagonal covariance. Off-diagonal entries are zero.";
    if (data.T > 1) {
        moments.note += " Waves are pooled under an invariant one-state model.";
    }
    if (!fitted.converged) {
        moments.note += " The EM did not converge.";
    }
    return moments;
}

}  // namespace lta
