#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lta/lta.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

int measurement_from(const std::string& name) {
    if (name == "categorical" || name == "lca") return lta::MEAS_CATEGORICAL;
    if (name == "gaussian" || name == "continuous" || name == "lpa") return lta::MEAS_GAUSSIAN;
    if (name == "mixed") return lta::MEAS_MIXED;
    throw std::invalid_argument(
        "measurement must be 'categorical', 'gaussian', or 'mixed'. "
        "Ordinal indicators are Release 2 and count families are later.");
}

// Copy a NumPy buffer in logical index order. A C-contiguous array of type T
// is one memcpy. A noncontiguous view is walked by its strides. A different
// dtype is cast first, which materializes one contiguous temporary.
template <typename T>
void copy_array(const py::handle& src, std::vector<T>& dest) {
    if (src.is_none()) return;
    if (!py::isinstance<py::array>(src)) {
        dest = src.cast<std::vector<T>>();
        return;
    }
    py::array array = py::reinterpret_borrow<py::array>(src);
    if (!py::isinstance<py::array_t<T>>(array)) {
        array = py::array_t<T, py::array::forcecast>::ensure(src);
        if (!array) throw std::invalid_argument("panel buffer has the wrong numeric type");
    }
    const auto info = array.request();
    dest.resize(static_cast<std::size_t>(info.size));
    const bool flat = info.ndim == 1 && info.strides[0] == static_cast<py::ssize_t>(sizeof(T));
    if (flat) {
        std::memcpy(dest.data(), info.ptr, static_cast<std::size_t>(info.size) * sizeof(T));
        return;
    }
    for (py::ssize_t n = 0; n < info.size; ++n) {
        const char* ptr = static_cast<const char*>(info.ptr);
        py::ssize_t rem = n;
        for (py::ssize_t axis = info.ndim - 1; axis >= 0; --axis) {
            const py::ssize_t len = info.shape[axis];
            const py::ssize_t coord = len == 0 ? 0 : rem % len;
            rem = len == 0 ? 0 : rem / len;
            ptr += coord * info.strides[axis];
        }
        dest[static_cast<std::size_t>(n)] = *reinterpret_cast<const T*>(ptr);
    }
}

lta::Panel panel_from(const py::dict& d) {
    lta::Panel p;
    p.N = d["N"].cast<int>();
    p.T = d["T"].cast<int>();
    p.J = d["J"].cast<int>();
    p.measurement = measurement_from(d["measurement"].cast<std::string>());
    if (d.contains("family") && !d["family"].is_none()) {
        for (auto item : d["family"].cast<std::vector<std::string>>())
            p.family.push_back(lta::parse_family_name(item));
    }
    if (d.contains("column_role") && !d["column_role"].is_none()) {
        p.column_role = d["column_role"].cast<std::vector<int>>();
    }
    if (d.contains("ncat") && !d["ncat"].is_none()) p.ncat = d["ncat"].cast<std::vector<int>>();
    p.Rmax = d.contains("Rmax") ? d["Rmax"].cast<int>() : 1;
    if (d.contains("cat") && !d["cat"].is_none()) copy_array<int>(d["cat"], p.cat);
    if (d.contains("gauss") && !d["gauss"].is_none()) copy_array<double>(d["gauss"], p.gauss);
    return p;
}

lta::Spec spec_from(const py::dict& d) {
    lta::Spec s;
    s.K = d["K"].cast<int>();
    if (d.contains("covariance") && !d["covariance"].is_none())
        s.cov_model = lta::parse_covariance_name(d["covariance"].cast<std::string>());
    s.measurement_invariant = d.contains("measurement_invariant") ? d["measurement_invariant"].cast<bool>() : true;
    s.means_invariant = d.contains("means_invariant") ? d["means_invariant"].cast<bool>() : true;
    s.covariance_invariant = d.contains("covariance_invariant") ? d["covariance_invariant"].cast<bool>() : true;
    s.transition_equal = d.contains("transition_equal") ? d["transition_equal"].cast<bool>() : true;
    s.n_starts = d.contains("n_starts") ? d["n_starts"].cast<int>() : 40;
    s.screen_iter = d.contains("screen_iter") ? d["screen_iter"].cast<int>() : 10;
    s.n_full = d.contains("n_full") ? d["n_full"].cast<int>() : 4;
    s.max_iter = d.contains("max_iter") ? d["max_iter"].cast<int>() : 200;
    s.tol = d.contains("tol") ? d["tol"].cast<double>() : 1e-7;
    s.replicate_tol = d.contains("replicate_tol") ? d["replicate_tol"].cast<double>() : 1e-3;
    s.n_threads = d.contains("n_threads") ? d["n_threads"].cast<int>() : 1;
    s.seed = d.contains("seed") ? d["seed"].cast<std::uint64_t>() : 1;
    s.align_labels = d.contains("align_labels") ? d["align_labels"].cast<bool>() : true;
    const bool has_start = d.contains("has_start") && d["has_start"].cast<bool>();
    const bool append_start = d.contains("append_supplied_start") && d["append_supplied_start"].cast<bool>();
    if ((has_start || append_start) && d.contains("start") && !d["start"].is_none()) {
        s.has_start = has_start;
        s.append_supplied_start = append_start && !has_start;
        auto st = d["start"].cast<py::dict>();
        if (st.contains("delta") && !st["delta"].is_none()) s.start.delta = st["delta"].cast<std::vector<double>>();
        if (st.contains("tau") && !st["tau"].is_none()) s.start.tau = st["tau"].cast<std::vector<double>>();
        if (st.contains("rho") && !st["rho"].is_none()) s.start.rho = st["rho"].cast<std::vector<double>>();
        if (st.contains("means") && !st["means"].is_none()) s.start.means = st["means"].cast<std::vector<double>>();
        if (st.contains("cov") && !st["cov"].is_none()) s.start.cov = st["cov"].cast<std::vector<double>>();
    }
    if (d.contains("covariates") && !d["covariates"].is_none()) {
        lta::CovariateSpec c;
        auto names = d["covariates"].cast<std::vector<std::string>>();
        c.names = names;
        c.on_initial = !names.empty();
        lta::check_covariate_spec(c);
    }
    return s;
}

py::dict params_to(const lta::Params& p) {
    py::dict d;
    d["delta"] = p.delta;
    d["tau"] = p.tau;
    d["rho"] = p.rho;
    d["means"] = p.means;
    d["cov"] = p.cov;
    return d;
}

py::dict fit_to(const lta::FitResult& r) {
    py::dict d;
    d["status"] = r.status == lta::STATUS_OK ? "ok" : r.status == lta::STATUS_WARNED ? "warned" : "unusable";
    d["status_reason"] = r.status_reason;
    d["engine"] = r.engine;
    d["schema_version"] = r.schema_version;
    d["loglik"] = r.loglik;
    d["converged"] = r.converged;
    d["regularized"] = r.regularized;
    d["ic_supported"] = r.ic_supported;
    d["n_iter"] = r.n_iter;
    d["n_starts"] = r.n_starts;
    d["n_full"] = r.n_full;
    d["n_converged_starts"] = r.n_converged_starts;
    d["n_best"] = r.n_best;
    d["n_parameters"] = r.n_parameters;
    d["n_subjects"] = r.n_subjects;
    d["n_analyzed"] = r.n_analyzed;
    d["n_excluded"] = r.n_excluded;
    d["n_waves"] = r.n_waves;
    d["n_indicators"] = r.n_indicators;
    d["K"] = r.K;
    d["aic"] = r.aic;
    d["bic"] = r.bic;
    d["sabic"] = r.sabic;
    d["entropy_wave"] = r.entropy_wave;
    d["entropy_sequence"] = r.entropy_sequence;
    d["entropy_wave_raw"] = r.entropy_wave_raw;
    d["entropy_sequence_raw"] = r.entropy_sequence_raw;
    d["min_occupancy"] = r.min_occupancy;
    d["min_eigenvalue"] = r.min_eigenvalue;
    d["cov_positive_definite"] = r.cov_positive_definite;
    d["cov_ill_conditioned"] = r.cov_ill_conditioned;
    d["cov_collapsed_component"] = r.cov_collapsed_component;
    d["condition_number"] = r.condition_number;
    d["min_relative_variance"] = r.min_relative_variance;
    d["n_numerically_empty"] = r.n_numerically_empty;
    d["n_empty_state_waves"] = r.n_empty_state_waves;
    d["n_negligible_transition_rows"] = r.n_negligible_transition_rows;
    d["eligible_model_comparison"] = r.eligible_model_comparison;
    d["eligible_assignment"] = r.eligible_assignment;
    d["eligible_measurement"] = r.eligible_measurement;
    d["eligible_transitions"] = r.eligible_transitions;
    d["eligible_further_starts"] = r.eligible_further_starts;
    d["eligibility_detail"] = r.eligibility_detail;
    d["viterbi_logprob"] = r.viterbi_logprob;
    d["n_boundary"] = r.n_boundary;
    d["seed"] = r.seed;
    d["n_threads"] = r.n_threads;
    d["tol"] = r.tol;
    d["replicate_tol"] = r.replicate_tol;
    d["data_hash"] = r.data_hash;
    d["excluded_subjects"] = r.excluded_subjects;
    d["params"] = params_to(r.params);
    d["n_tau_blocks"] = r.n_tau_blocks;
    d["n_meas_blocks"] = r.n_meas_blocks;
    d["n_mean_blocks"] = r.n_mean_blocks;
    d["n_cov_blocks"] = r.n_cov_blocks;
    d["Rmax"] = r.Rmax;
    d["measurement"] = r.measurement;
    d["family"] = r.family;
    d["gaussian_index"] = r.gaussian_index;
    d["categorical_index"] = r.categorical_index;
    d["n_gaussian"] = r.n_gaussian;
    d["n_categorical"] = r.n_categorical;
    d["covariance"] = r.covariance_name;
    d["measurement_invariant"] = r.measurement_invariant;
    d["means_invariant"] = r.means_invariant;
    d["covariance_invariant"] = r.covariance_invariant;
    d["transition_equal"] = r.transition_equal;
    d["gamma"] = r.gamma;
    d["filtered"] = r.filtered;
    d["xi"] = r.xi;
    d["viterbi"] = r.viterbi;
    d["modal"] = r.modal;
    d["label_order"] = r.label_order;
    d["prevalence_model"] = r.prevalence_model;
    d["prevalence_posterior"] = r.prevalence_posterior;
    d["prevalence_hard"] = r.prevalence_hard;
    d["expected_membership"] = r.expected_membership;
    d["expected_transitions"] = r.expected_transitions;
    d["hard_transitions"] = r.hard_transitions;
    d["start_id"] = r.start_id;
    d["start_phase"] = r.start_phase;
    d["start_loglik"] = r.start_loglik;
    d["start_iters"] = r.start_iters;
    d["start_converged"] = r.start_converged;
    d["start_selected"] = r.start_selected;
    d["start_admissible"] = r.start_admissible;
    d["admissible"] = r.admissible;
    d["rejected_start"] = r.rejected_start;
    d["rejected_loglik"] = r.rejected_loglik;
    d["rejected_n"] = r.rejected_n;
    d["rejected_level"] = r.rejected_level;
    d["rejected_reason"] = r.rejected_reason;
    d["rejected_delta"] = r.rejected_delta;
    d["rejected_means"] = r.rejected_means;
    d["rejected_cov"] = r.rejected_cov;
    d["rejected_class_eigen"] = r.rejected_class_eigen;
    d["rejected_min_class_n"] = r.rejected_min_class_n;
    d["trace_iter"] = r.trace_iter;
    d["trace_ll"] = r.trace_ll;
    d["has_draws"] = r.has_draws;
    d["n_draws"] = r.n_draws;
    d["draws"] = r.draws;
    d["parameter_uncertainty"] = r.parameter_uncertainty;
    d["state_uncertainty"] = r.state_uncertainty;
    d["interval_type"] = r.interval_type;
    d["miss_indicator_n"] = r.miss_indicator_n;
    d["miss_indicator_rate"] = r.miss_indicator_rate;
    d["miss_wave_n"] = r.miss_wave_n;
    d["miss_wave_rate"] = r.miss_wave_rate;
    d["miss_pattern"] = r.miss_pattern;
    d["miss_pattern_n"] = r.miss_pattern_n;
    d["n_monotone"] = r.n_monotone;
    d["n_intermittent"] = r.n_intermittent;
    d["n_complete_subjects"] = r.n_complete_subjects;
    return d;
}

lta::FitResult fit_from(const py::dict& d) {
    lta::FitResult r;
    auto st = d["status"].cast<std::string>();
    r.status = st == "ok" ? lta::STATUS_OK : st == "warned" ? lta::STATUS_WARNED : lta::STATUS_UNUSABLE;
    r.status_reason = d["status_reason"].cast<std::string>();
    r.engine = d["engine"].cast<std::string>();
    r.loglik = d["loglik"].cast<double>();
    r.converged = d["converged"].cast<bool>();
    r.n_parameters = d["n_parameters"].cast<int>();
    r.aic = d["aic"].cast<double>();
    r.bic = d["bic"].cast<double>();
    r.n_best = d["n_best"].cast<int>();
    r.n_starts = d["n_starts"].cast<int>();
    r.K = d["K"].cast<int>();
    r.n_waves = d["n_waves"].cast<int>();
    r.n_indicators = d["n_indicators"].cast<int>();
    r.data_hash = d["data_hash"].cast<std::string>();
    r.measurement = d["measurement"].cast<int>();
    r.family = d["family"].cast<std::vector<int>>();
    r.n_gaussian = d["n_gaussian"].cast<int>();
    r.n_categorical = d["n_categorical"].cast<int>();
    r.measurement_invariant = d["measurement_invariant"].cast<bool>();
    r.means_invariant = d["means_invariant"].cast<bool>();
    r.covariance_invariant = d["covariance_invariant"].cast<bool>();
    r.transition_equal = d["transition_equal"].cast<bool>();
    r.n_tau_blocks = d["n_tau_blocks"].cast<int>();
    r.n_meas_blocks = d["n_meas_blocks"].cast<int>();
    r.n_mean_blocks = d["n_mean_blocks"].cast<int>();
    r.n_cov_blocks = d["n_cov_blocks"].cast<int>();
    r.Rmax = d["Rmax"].cast<int>();
    r.covariance_name = d["covariance"].cast<std::string>();
    if (!r.covariance_name.empty()) r.cov_model = lta::parse_covariance_name(r.covariance_name);
    r.ic_supported = d["ic_supported"].cast<bool>();
    auto p = d["params"].cast<py::dict>();
    r.params.delta = p["delta"].cast<std::vector<double>>();
    r.params.tau = p["tau"].cast<std::vector<double>>();
    r.params.rho = p["rho"].cast<std::vector<double>>();
    r.params.means = p["means"].cast<std::vector<double>>();
    r.params.cov = p["cov"].cast<std::vector<double>>();
    auto take_d = [&](const char* name, std::vector<double>& dest) {
        if (d.contains(name) && !d[name].is_none()) dest = d[name].cast<std::vector<double>>();
    };
    auto take_i = [&](const char* name, std::vector<int>& dest) {
        if (d.contains(name) && !d[name].is_none()) dest = d[name].cast<std::vector<int>>();
    };
    take_d("gamma", r.gamma);
    take_d("filtered", r.filtered);
    take_d("xi", r.xi);
    take_i("viterbi", r.viterbi);
    take_i("modal", r.modal);
    take_i("label_order", r.label_order);
    take_d("prevalence_model", r.prevalence_model);
    take_d("prevalence_posterior", r.prevalence_posterior);
    take_d("prevalence_hard", r.prevalence_hard);
    take_d("expected_membership", r.expected_membership);
    take_d("expected_transitions", r.expected_transitions);
    take_d("hard_transitions", r.hard_transitions);
    take_i("start_id", r.start_id);
    take_i("start_phase", r.start_phase);
    take_d("start_loglik", r.start_loglik);
    take_i("start_iters", r.start_iters);
    take_i("start_converged", r.start_converged);
    take_i("start_selected", r.start_selected);
    take_i("start_admissible", r.start_admissible);
    take_i("rejected_start", r.rejected_start);
    take_i("rejected_n", r.rejected_n);
    take_d("rejected_loglik", r.rejected_loglik);
    take_d("rejected_delta", r.rejected_delta);
    take_d("rejected_means", r.rejected_means);
    take_d("rejected_cov", r.rejected_cov);
    take_d("rejected_class_eigen", r.rejected_class_eigen);
    take_d("rejected_min_class_n", r.rejected_min_class_n);
    if (d.contains("rejected_level") && !d["rejected_level"].is_none())
        r.rejected_level = d["rejected_level"].cast<std::vector<std::string>>();
    if (d.contains("rejected_reason") && !d["rejected_reason"].is_none())
        r.rejected_reason = d["rejected_reason"].cast<std::vector<std::string>>();
    take_i("trace_iter", r.trace_iter);
    take_d("trace_ll", r.trace_ll);
    if (d.contains("gaussian_index") && !d["gaussian_index"].is_none())
        r.gaussian_index = d["gaussian_index"].cast<std::vector<int>>();
    if (d.contains("categorical_index") && !d["categorical_index"].is_none())
        r.categorical_index = d["categorical_index"].cast<std::vector<int>>();
    r.schema_version = 2;
    r.interval_type = "none";
    r.parameter_uncertainty = "none";
    r.state_uncertainty = "plugin_posterior";
    if (d.contains("n_iter")) r.n_iter = d["n_iter"].cast<int>();
    if (d.contains("n_full")) r.n_full = d["n_full"].cast<int>();
    if (d.contains("n_converged_starts")) r.n_converged_starts = d["n_converged_starts"].cast<int>();
    if (d.contains("regularized")) r.regularized = d["regularized"].cast<bool>();
    if (d.contains("sabic")) r.sabic = d["sabic"].cast<double>();
    if (d.contains("entropy_wave")) r.entropy_wave = d["entropy_wave"].cast<double>();
    if (d.contains("entropy_sequence")) r.entropy_sequence = d["entropy_sequence"].cast<double>();
    if (d.contains("min_eigenvalue") && !d["min_eigenvalue"].is_none())
        r.min_eigenvalue = d["min_eigenvalue"].cast<double>();
    if (d.contains("n_boundary")) r.n_boundary = d["n_boundary"].cast<int>();
    if (d.contains("n_subjects")) r.n_subjects = d["n_subjects"].cast<int>();
    if (d.contains("n_analyzed")) r.n_analyzed = d["n_analyzed"].cast<int>();
    if (d.contains("cov_positive_definite") && !d["cov_positive_definite"].is_none())
        r.cov_positive_definite = d["cov_positive_definite"].cast<bool>();
    if (d.contains("cov_collapsed_component") && !d["cov_collapsed_component"].is_none())
        r.cov_collapsed_component = d["cov_collapsed_component"].cast<bool>();
    if (d.contains("admissible") && !d["admissible"].is_none())
        r.admissible = d["admissible"].cast<bool>();
    else
        r.admissible = r.converged && std::isfinite(r.loglik) && !r.regularized &&
                       r.cov_positive_definite && !r.cov_collapsed_component;
    if (d.contains("schema_version") && !d["schema_version"].is_none())
        r.schema_version = d["schema_version"].cast<int>();
    if (d.contains("entropy_wave_raw") && !d["entropy_wave_raw"].is_none())
        r.entropy_wave_raw = d["entropy_wave_raw"].cast<double>();
    if (d.contains("entropy_sequence_raw") && !d["entropy_sequence_raw"].is_none())
        r.entropy_sequence_raw = d["entropy_sequence_raw"].cast<double>();
    if (d.contains("min_occupancy") && !d["min_occupancy"].is_none())
        r.min_occupancy = d["min_occupancy"].cast<double>();
    if (d.contains("condition_number") && !d["condition_number"].is_none())
        r.condition_number = d["condition_number"].cast<double>();
    if (d.contains("min_relative_variance") && !d["min_relative_variance"].is_none())
        r.min_relative_variance = d["min_relative_variance"].cast<double>();
    if (d.contains("n_numerically_empty") && !d["n_numerically_empty"].is_none())
        r.n_numerically_empty = d["n_numerically_empty"].cast<int>();
    if (d.contains("n_empty_state_waves") && !d["n_empty_state_waves"].is_none())
        r.n_empty_state_waves = d["n_empty_state_waves"].cast<int>();
    if (d.contains("n_negligible_transition_rows") && !d["n_negligible_transition_rows"].is_none())
        r.n_negligible_transition_rows = d["n_negligible_transition_rows"].cast<int>();
    if (d.contains("cov_ill_conditioned") && !d["cov_ill_conditioned"].is_none())
        r.cov_ill_conditioned = d["cov_ill_conditioned"].cast<bool>();
    if (d.contains("eligible_model_comparison") && !d["eligible_model_comparison"].is_none())
        r.eligible_model_comparison = d["eligible_model_comparison"].cast<bool>();
    if (d.contains("eligible_assignment") && !d["eligible_assignment"].is_none())
        r.eligible_assignment = d["eligible_assignment"].cast<bool>();
    if (d.contains("eligible_measurement") && !d["eligible_measurement"].is_none())
        r.eligible_measurement = d["eligible_measurement"].cast<bool>();
    if (d.contains("eligible_transitions") && !d["eligible_transitions"].is_none())
        r.eligible_transitions = d["eligible_transitions"].cast<bool>();
    if (d.contains("eligible_further_starts") && !d["eligible_further_starts"].is_none())
        r.eligible_further_starts = d["eligible_further_starts"].cast<bool>();
    if (d.contains("eligibility_detail") && !d["eligibility_detail"].is_none())
        r.eligibility_detail = d["eligibility_detail"].cast<std::string>();
    take_d("viterbi_logprob", r.viterbi_logprob);
    if (d.contains("seed") && !d["seed"].is_none()) r.seed = d["seed"].cast<std::uint64_t>();
    if (d.contains("n_threads") && !d["n_threads"].is_none()) r.n_threads = d["n_threads"].cast<int>();
    if (d.contains("tol") && !d["tol"].is_none()) r.tol = d["tol"].cast<double>();
    if (d.contains("replicate_tol") && !d["replicate_tol"].is_none())
        r.replicate_tol = d["replicate_tol"].cast<double>();
    take_i("excluded_subjects", r.excluded_subjects);
    if (d.contains("has_draws") && !d["has_draws"].is_none()) r.has_draws = d["has_draws"].cast<bool>();
    if (d.contains("n_draws") && !d["n_draws"].is_none()) r.n_draws = d["n_draws"].cast<int>();
    take_d("draws", r.draws);
    if (d.contains("parameter_uncertainty") && !d["parameter_uncertainty"].is_none())
        r.parameter_uncertainty = d["parameter_uncertainty"].cast<std::string>();
    if (d.contains("state_uncertainty") && !d["state_uncertainty"].is_none())
        r.state_uncertainty = d["state_uncertainty"].cast<std::string>();
    if (d.contains("interval_type") && !d["interval_type"].is_none())
        r.interval_type = d["interval_type"].cast<std::string>();
    take_d("miss_indicator_n", r.miss_indicator_n);
    take_d("miss_indicator_rate", r.miss_indicator_rate);
    take_d("miss_wave_n", r.miss_wave_n);
    take_d("miss_wave_rate", r.miss_wave_rate);
    if (d.contains("miss_pattern") && !d["miss_pattern"].is_none())
        r.miss_pattern = d["miss_pattern"].cast<std::vector<std::string>>();
    take_i("miss_pattern_n", r.miss_pattern_n);
    if (d.contains("n_monotone") && !d["n_monotone"].is_none()) r.n_monotone = d["n_monotone"].cast<int>();
    if (d.contains("n_intermittent") && !d["n_intermittent"].is_none())
        r.n_intermittent = d["n_intermittent"].cast<int>();
    if (d.contains("n_complete_subjects") && !d["n_complete_subjects"].is_none())
        r.n_complete_subjects = d["n_complete_subjects"].cast<int>();
    if (d.contains("n_excluded") && !d["n_excluded"].is_none()) r.n_excluded = d["n_excluded"].cast<int>();
    return r;
}

lta::Params params_from(const py::dict& p) {
    lta::Params out;
    if (p.contains("delta") && !p["delta"].is_none()) out.delta = p["delta"].cast<std::vector<double>>();
    if (p.contains("tau") && !p["tau"].is_none()) out.tau = p["tau"].cast<std::vector<double>>();
    if (p.contains("rho") && !p["rho"].is_none()) out.rho = p["rho"].cast<std::vector<double>>();
    if (p.contains("means") && !p["means"].is_none()) out.means = p["means"].cast<std::vector<double>>();
    if (p.contains("cov") && !p["cov"].is_none()) out.cov = p["cov"].cast<std::vector<double>>();
    return out;
}

py::dict alignment_to(const lta::Alignment& a) {
    py::dict d;
    d["permutation"] = a.permutation;
    d["cost"] = a.cost;
    d["runner_up_cost"] = a.runner_up_cost;
    d["ambiguous"] = a.ambiguous;
    d["distance"] = a.distance;
    d["note"] = a.note;
    return d;
}

py::dict compare_to(const lta::CompareResult& c) {
    py::dict d;
    d["refused"] = c.refused;
    d["reason"] = c.reason;
    d["warnings"] = c.warnings;
    d["lr_statistic"] = c.lr_statistic;
    d["p_value_available"] = c.p_value_available;
    py::list models;
    for (const auto& m : c.models) {
        py::dict row;
        row["role"] = m.role;
        row["loglik"] = m.loglik;
        row["n_parameters"] = m.n_parameters;
        row["aic"] = m.aic;
        row["bic"] = m.bic;
        row["converged"] = m.converged;
        row["n_best"] = m.n_best;
        row["n_starts"] = m.n_starts;
        row["status"] = m.status;
        row["ic_supported"] = m.ic_supported;
        row["note"] = m.note;
        models.append(row);
    }
    d["models"] = models;
    py::list diffs;
    for (const auto& m : c.differences) {
        py::dict row;
        row["parameter"] = m.parameter;
        row["state"] = m.state;
        row["wave"] = m.wave;
        row["variable"] = m.variable;
        row["category"] = m.category;
        row["estimate_a"] = m.estimate_a;
        row["estimate_b"] = m.estimate_b;
        row["difference"] = m.difference;
        diffs.append(row);
    }
    d["differences"] = diffs;
    return d;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Shared C++ EM core for pyfylr. Release 1. engine is cpp-em; draws are empty.";

    m.def("build_info", []() {
        auto b = lta::build_info();
        py::dict d;
        d["eigen_version"] = b.eigen_version;
        d["compiler"] = b.compiler;
        d["flags"] = b.flags;
        d["cxx_standard"] = b.cxx_standard;
        d["eigen_dont_parallelize"] = b.eigen_dont_parallelize;
        d["external_blas"] = b.external_blas;
        return d;
    });

    m.def("fit", [](py::dict panel, py::dict spec) {
        return fit_to(lta::fit(panel_from(panel), spec_from(spec), nullptr));
    });
    m.def("blrt_hardware_threads", []() { return lta::blrt_hardware_threads(); });
    m.def("blrt_fit_draws", [](py::list panels, std::vector<int> present, py::dict h0, py::dict h1,
                              std::uint64_t base, int max_retries, int n_threads) {
        if (static_cast<int>(panels.size()) != static_cast<int>(present.size()))
            throw std::invalid_argument("panels and present must have the same length");
        const auto convert_t0 = std::chrono::steady_clock::now();
        std::vector<lta::BlrtDrawJob> jobs(panels.size());
        lta::Spec spec0 = spec_from(h0);
        lta::Spec spec1 = spec_from(h1);
        for (size_t i = 0; i < jobs.size(); ++i) {
            if (!present[i]) continue;
            if (panels[i].is_none()) throw std::invalid_argument("a present draw needs a panel");
            jobs[i].present = true;
            jobs[i].panel = panel_from(panels[i].cast<py::dict>());
        }
        const double convert_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - convert_t0).count();
        std::vector<lta::BlrtDrawFit> rows;
        std::vector<lta::WorkerLoad> loads;
        const auto engine_t0 = std::chrono::steady_clock::now();
        {
            py::gil_scoped_release release;
            rows = lta::blrt_fit_draws(jobs, spec0, spec1, base, max_retries, n_threads, nullptr, &loads);
        }
        const double engine_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - engine_t0).count();
        const auto export_t0 = std::chrono::steady_clock::now();
        py::dict out;
        py::list ok, reason, lr, h0_replicated, h1_replicated, h1_best_fit_replicated, h1_replication_assessed;
        py::list attempt_draw, attempt_retry, attempt_starts, attempt_starts_h0, attempt_starts_h1, attempt_h1_embedded;
        for (const auto& row : rows) {
            ok.append(row.ok);
            reason.append(row.reason);
            lr.append(row.ok ? row.lr : std::numeric_limits<double>::quiet_NaN());
            h0_replicated.append(row.h0_replicated);
            h1_replicated.append(row.h1_replicated);
            h1_best_fit_replicated.append(row.h1_best_fit_replicated);
            h1_replication_assessed.append(row.h1_replication_assessed);
            for (const auto& attempt : row.attempts) {
                attempt_draw.append(attempt.draw);
                attempt_retry.append(attempt.retry);
                attempt_starts.append(attempt.n_starts);
                attempt_starts_h0.append(attempt.n_starts_h0);
                attempt_starts_h1.append(attempt.n_starts_h1);
                attempt_h1_embedded.append(attempt.h1_embedded);
            }
        }
        int used = std::min(n_threads, lta::blrt_hardware_threads());
        if (!rows.empty()) used = std::min(used, static_cast<int>(rows.size()));
        if (used < 1) used = 1;
        out["ok"] = ok;
        out["reason"] = reason;
        out["lr"] = lr;
        out["h0_replicated"] = h0_replicated;
        out["h1_replicated"] = h1_replicated;
        out["h1_best_fit_replicated"] = h1_best_fit_replicated;
        out["h1_replication_assessed"] = h1_replication_assessed;
        out["n_threads_used"] = used;
        out["attempt_draw"] = attempt_draw;
        out["attempt_retry"] = attempt_retry;
        out["attempt_starts"] = attempt_starts;
        out["attempt_starts_h0"] = attempt_starts_h0;
        out["attempt_starts_h1"] = attempt_starts_h1;
        out["attempt_h1_embedded"] = attempt_h1_embedded;
        py::list worker_jobs, worker_busy;
        for (const auto& slot : loads) {
            worker_jobs.append(slot.jobs);
            worker_busy.append(slot.busy_seconds);
        }
        out["worker_jobs"] = worker_jobs;
        out["worker_busy_s"] = worker_busy;
        out["seconds_convert"] = convert_s;
        out["seconds_engine"] = engine_s;
        out["seconds_export"] =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - export_t0).count();
        return out;
    });
    m.def("split_null_state", [](py::dict params, int k0, int k1, const std::string& covariance, int n_tau,
                                 int n_meas, int n_mean, int n_cov, int n_gaussian, int J, int Rmax) {
        lta::Params src;
        if (params.contains("delta") && !params["delta"].is_none())
            src.delta = params["delta"].cast<std::vector<double>>();
        if (params.contains("tau") && !params["tau"].is_none()) src.tau = params["tau"].cast<std::vector<double>>();
        if (params.contains("rho") && !params["rho"].is_none()) src.rho = params["rho"].cast<std::vector<double>>();
        if (params.contains("means") && !params["means"].is_none())
            src.means = params["means"].cast<std::vector<double>>();
        if (params.contains("cov") && !params["cov"].is_none()) src.cov = params["cov"].cast<std::vector<double>>();
        lta::NullSplitLayout layout;
        layout.cov_model = lta::parse_covariance_name(covariance);
        layout.n_tau = n_tau;
        layout.n_meas = n_meas;
        layout.n_mean = n_mean;
        layout.n_cov = n_cov;
        layout.n_gaussian = n_gaussian;
        layout.J = J;
        layout.Rmax = Rmax;
        lta::Params dest;
        if (!lta::blrt_split_null_state(src, k0, k1, layout, dest)) return py::object(py::none());
        return py::object(params_to(dest));
    });
    m.def("add_starts", [](py::dict panel, py::dict spec, py::dict previous, int n_new, double seed) {
        auto fit = lta::add_starts(panel_from(panel), spec_from(spec), fit_from(previous), n_new,
                                   static_cast<std::uint64_t>(seed));
        return fit_to(fit);
    });
    m.def("align_states", [](py::dict panel, py::dict spec, py::dict fit, py::dict reference) {
        lta::Alignment info;
        auto aligned = lta::align_fit(panel_from(panel), spec_from(spec), fit_from(fit), params_from(reference),
                                      &info);
        py::dict out;
        out["result"] = fit_to(aligned);
        out["alignment"] = alignment_to(info);
        return out;
    });
    m.def("evaluate", [](py::dict panel, py::dict spec, py::dict params) {
        return fit_to(lta::evaluate(panel_from(panel), spec_from(spec), params_from(params)));
    });
    m.def("subject_loglik", [](py::dict panel, py::dict spec, py::dict params) {
        return lta::subject_loglik(panel_from(panel), spec_from(spec), params_from(params));
    });
    m.def("observed_loglik", [](py::dict panel, py::dict spec, py::dict params) {
        lta::Params p;
        p.delta = params["delta"].cast<std::vector<double>>();
        p.tau = params["tau"].cast<std::vector<double>>();
        if (params.contains("rho")) p.rho = params["rho"].cast<std::vector<double>>();
        if (params.contains("means")) p.means = params["means"].cast<std::vector<double>>();
        if (params.contains("cov")) p.cov = params["cov"].cast<std::vector<double>>();
        return lta::observed_loglik(panel_from(panel), spec_from(spec), p);
    });
    m.def("describe_indicators", [](py::dict panel, bool by_wave) {
        auto rows = lta::describe_indicators(panel_from(panel), by_wave);
        py::list out;
        for (const auto& r : rows) {
            py::dict d;
            d["variable"] = r.variable;
            d["wave"] = r.wave;
            d["category"] = r.category;
            d["n"] = r.n;
            d["n_missing"] = r.n_missing;
            d["mean"] = r.mean;
            d["sd"] = r.sd;
            d["median"] = r.median;
            d["q25"] = r.q25;
            d["q75"] = r.q75;
            d["min"] = r.min;
            d["max"] = r.max;
            d["skew"] = r.skew;
            d["kurtosis"] = r.kurtosis;
            d["proportion"] = r.proportion;
            d["kind"] = r.kind;
            out.append(d);
        }
        return out;
    });
    m.def("describe_missingness", [](py::dict panel) {
        auto s = lta::describe_missingness(panel_from(panel));
        py::dict d;
        d["n_subjects"] = s.n_subjects;
        d["n_excluded"] = s.n_excluded;
        d["n_complete"] = s.n_complete;
        d["n_monotone"] = s.n_monotone;
        d["n_intermittent"] = s.n_intermittent;
        d["n_complete_indicators"] = s.n_complete_indicators;
        d["n_monotone_dropout"] = s.n_monotone_dropout;
        d["excluded_subjects"] = s.excluded_subjects;
        d["indicator_n_missing"] = s.indicator_n_missing;
        d["indicator_rate"] = s.indicator_rate;
        d["wave_n_fully_missing"] = s.wave_n_fully_missing;
        d["wave_rate"] = s.wave_rate;
        d["pattern"] = s.pattern;
        d["pattern_n"] = s.pattern_n;
        d["note"] = s.note;
        return d;
    });
    m.def("standardize_indicators", [](py::dict panel, const std::string& scope) {
        auto s = lta::standardize_indicators(panel_from(panel), scope);
        py::dict d;
        d["values"] = s.values;
        d["center"] = s.center;
        d["scale"] = s.scale;
        d["scope"] = s.scope;
        d["warned"] = s.warned;
        d["warning"] = s.warning;
        return d;
    });
    m.def("describe_profiles", [](py::dict panel, py::dict spec, py::dict fit, py::object reference_sd) {
        std::vector<double> ref;
        if (!reference_sd.is_none()) ref = reference_sd.cast<std::vector<double>>();
        auto rows = lta::describe_profiles(panel_from(panel), spec_from(spec), fit_from(fit), ref);
        py::list out;
        for (const auto& r : rows) {
            py::dict d;
            d["variable"] = r.variable;
            d["state"] = r.state;
            d["wave"] = r.wave;
            d["category"] = r.category;
            d["source"] = r.source;
            d["statistic"] = r.statistic;
            d["estimate"] = r.estimate;
            d["weight"] = r.weight;
            d["reference_value"] = r.reference_value;
            d["reference_sd"] = r.reference_sd;
            d["difference"] = r.difference;
            d["standardized_difference"] = r.standardized_difference;
            d["reference_sd_source"] = r.reference_sd_source;
            d["weighting"] = r.weighting;
            out.append(d);
        }
        return out;
    });
    m.def("compare_invariance", [](py::dict panel, py::dict restricted, py::dict unrestricted) {
        return compare_to(lta::compare_invariance(panel_from(panel), fit_from(restricted), fit_from(unrestricted)));
    });
    m.def("compare_transitions", [](py::dict panel, py::dict equal_fit, py::dict free_fit) {
        return compare_to(lta::compare_transitions(panel_from(panel), fit_from(equal_fit), fit_from(free_fit)));
    });
    m.def("parse_family", [](const std::string& name) { return lta::family_name(lta::parse_family_name(name)); });
    m.def("parse_covariance", [](const std::string& name) { return lta::covariance_name(lta::parse_covariance_name(name)); });
    m.def("estimate_distal", [](py::dict spec) {
        lta::DistalSpec s;
        s.name = spec["name"].cast<std::string>();
        auto family = spec["family"].cast<std::string>();
        if (family == "continuous") s.family = lta::OUT_CONTINUOUS;
        else if (family == "binary") s.family = lta::OUT_BINARY;
        else if (family == "nominal") s.family = lta::OUT_NOMINAL;
        else if (family == "ordinal") s.family = lta::OUT_ORDINAL;
        else if (family == "count" || family == "poisson") s.family = lta::OUT_COUNT;
        else throw std::invalid_argument("unknown distal outcome family '" + family + "'");
        auto timing = spec.contains("timing") && !spec["timing"].is_none()
                          ? spec["timing"].cast<std::string>()
                          : std::string();
        if (timing == "initial") s.timing = lta::OUTCOME_INITIAL;
        else if (timing == "final") s.timing = lta::OUTCOME_FINAL;
        else if (timing == "wave") s.timing = lta::OUTCOME_WAVE;
        else if (timing == "transition") s.timing = lta::OUTCOME_TRANSITION;
        else s.timing = -1;
        if (spec.contains("wave") && !spec["wave"].is_none()) s.wave = spec["wave"].cast<int>();
        if (spec.contains("from_state") && !spec["from_state"].is_none())
            s.from_state = spec["from_state"].cast<int>();
        if (spec.contains("to_state") && !spec["to_state"].is_none())
            s.to_state = spec["to_state"].cast<int>();
        if (spec.contains("method") && !spec["method"].is_none())
            s.method = spec["method"].cast<std::string>();
        lta::estimate_distal(s);
        return py::none();
    });
    m.def("correlate_indicators", [](py::dict panel) {
        auto rows = lta::correlate_indicators(panel_from(panel));
        py::list out;
        for (const auto& r : rows) {
            py::dict d;
            d["variable_a"] = r.variable_a;
            d["variable_b"] = r.variable_b;
            d["wave"] = r.wave;
            d["n"] = r.n;
            d["correlation"] = r.correlation;
            d["covariance"] = r.covariance;
            d["method"] = r.method;
            out.append(d);
        }
        return out;
    });
    m.def("fiml_moments", [](py::dict panel, const std::string& covariance) {
        auto moments = lta::fiml_moments(panel_from(panel), covariance);
        py::dict d;
        d["covariance"] = moments.covariance;
        d["n"] = moments.n;
        d["p"] = moments.p;
        d["loglik"] = moments.loglik;
        d["n_parameters"] = moments.n_parameters;
        d["converged"] = moments.converged;
        d["mean"] = moments.mean;
        d["cov"] = moments.cov;
        d["cor"] = moments.cor;
        d["note"] = moments.note;
        return d;
    });
}
