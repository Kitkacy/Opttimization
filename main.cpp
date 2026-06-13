#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct Edge {
    int u;
    int v;
    int weight;
};

struct Graph {
    int num_vertices = 0;
    std::vector<Edge> edges;
    std::vector<std::vector<std::pair<int, int>>> adjacency;
};

struct NodeRanking {
    std::vector<int> order;
    std::vector<int> position;
};

struct GreedyByNodeHeuristic {
    std::vector<int> greedy_solution;
    int preserved_node_count = 0;
};

struct RankedMutationProfile {
    std::vector<double> per_node_rate;
    double min_rate = 0.0;
    double max_rate = 0.0;
};

struct FitnessStats {
    long long evaluations = 0;
};

struct Individual {
    std::vector<int> genes;
    long long fitness = 0;
    bool evaluated = false;
};

struct GAConfig {
    int population_size = 100;
    long long nfe_budget = 50000;
    int max_generations = 0;
    double crossover_rate = 0.8;
    int tournament_size = 3;
    int elitism_count = 2;
    double mutation_rate = 0.0;
    bool auto_mutation_rate = true;
    unsigned int random_seed = 0;
    bool greedy_by_node_enabled = false;
    double greedy_by_node_repair_rate = 0.5;
    double greedy_by_node_preserve_fraction = 0.2;
    bool ranked_mutation_enabled = false;
    double ranked_mutation_min_multiplier = 0.25;
    double ranked_mutation_max_multiplier = 2.0;
    bool ranked_mutation_normalize = true;
    bool balancer_enabled = false;
    double balancer_min_fraction = 0.4;
    double balancer_max_fraction = 0.6;
};

struct GAResult {
    std::vector<int> best_solution;
    long long best_fitness = 0;
    long long fitness_evaluations = 0;
    int generations_run = 0;
    double elapsed_ms = 0.0;
    std::string stop_reason;
};

std::string trim(const std::string& text) {
    size_t start = 0;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) {
        ++start;
    }

    size_t end = text.size();
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }

    return text.substr(start, end - start);
}

bool parse_int(const std::string& value, int& out) {
    try {
        size_t consumed = 0;
        const long long parsed = std::stoll(value, &consumed);
        if (consumed != value.size() || parsed < std::numeric_limits<int>::min() ||
            parsed > std::numeric_limits<int>::max()) {
            return false;
        }
        out = static_cast<int>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_long_long(const std::string& value, long long& out) {
    try {
        size_t consumed = 0;
        out = std::stoll(value, &consumed);
        return consumed == value.size();
    } catch (...) {
        return false;
    }
}

bool parse_double(const std::string& value, double& out) {
    try {
        size_t consumed = 0;
        out = std::stod(value, &consumed);
        return consumed == value.size();
    } catch (...) {
        return false;
    }
}

bool parse_bool(const std::string& value, bool& out) {
    const std::string normalized = trim(value);
    if (normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on") {
        out = true;
        return true;
    }
    if (normalized == "0" || normalized == "false" || normalized == "no" || normalized == "off") {
        out = false;
        return true;
    }
    return false;
}

bool parse_unsigned_int(const std::string& value, unsigned int& out) {
    try {
        size_t consumed = 0;
        const unsigned long parsed = std::stoul(value, &consumed);
        if (consumed != value.size() || parsed > std::numeric_limits<unsigned int>::max()) {
            return false;
        }
        out = static_cast<unsigned int>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool load_config(const std::string& path, GAConfig& config) {
    std::ifstream input(path);
    if (!input) {
        std::cerr << "Error: could not open config \"" << path << "\"\n";
        return false;
    }

    std::unordered_map<std::string, std::string> values;
    std::string line;
    int line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;
        const size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const size_t separator_pos = line.find('=');
        if (separator_pos == std::string::npos) {
            std::cerr << "Error: invalid config line " << line_number << " in \"" << path
                      << "\" (expected key = value)\n";
            return false;
        }

        const std::string key = trim(line.substr(0, separator_pos));
        const std::string value = trim(line.substr(separator_pos + 1));
        if (key.empty() || value.empty()) {
            std::cerr << "Error: empty key or value on line " << line_number << " in \"" << path
                      << "\"\n";
            return false;
        }

        values[key] = value;
    }

    for (const auto& [key, value] : values) {
        if (key == "population_size") {
            if (!parse_int(value, config.population_size)) {
                std::cerr << "Error: invalid population_size \"" << value << "\"\n";
                return false;
            }
        } else if (key == "nfe_budget") {
            if (!parse_long_long(value, config.nfe_budget)) {
                std::cerr << "Error: invalid nfe_budget \"" << value << "\"\n";
                return false;
            }
        } else if (key == "max_generations") {
            if (!parse_int(value, config.max_generations)) {
                std::cerr << "Error: invalid max_generations \"" << value << "\"\n";
                return false;
            }
        } else if (key == "crossover_rate") {
            if (!parse_double(value, config.crossover_rate)) {
                std::cerr << "Error: invalid crossover_rate \"" << value << "\"\n";
                return false;
            }
        } else if (key == "tournament_size") {
            if (!parse_int(value, config.tournament_size)) {
                std::cerr << "Error: invalid tournament_size \"" << value << "\"\n";
                return false;
            }
        } else if (key == "elitism_count") {
            if (!parse_int(value, config.elitism_count)) {
                std::cerr << "Error: invalid elitism_count \"" << value << "\"\n";
                return false;
            }
        } else if (key == "mutation_rate") {
            if (!parse_double(value, config.mutation_rate)) {
                std::cerr << "Error: invalid mutation_rate \"" << value << "\"\n";
                return false;
            }
            config.auto_mutation_rate = (config.mutation_rate == 0.0);
        } else if (key == "random_seed") {
            if (!parse_unsigned_int(value, config.random_seed)) {
                std::cerr << "Error: invalid random_seed \"" << value << "\"\n";
                return false;
            }
        } else if (key == "greedy_by_node_enabled") {
            if (!parse_bool(value, config.greedy_by_node_enabled)) {
                std::cerr << "Error: invalid greedy_by_node_enabled \"" << value << "\"\n";
                return false;
            }
        } else if (key == "greedy_by_node_repair_rate") {
            if (!parse_double(value, config.greedy_by_node_repair_rate)) {
                std::cerr << "Error: invalid greedy_by_node_repair_rate \"" << value << "\"\n";
                return false;
            }
        } else if (key == "greedy_by_node_preserve_fraction") {
            if (!parse_double(value, config.greedy_by_node_preserve_fraction)) {
                std::cerr << "Error: invalid greedy_by_node_preserve_fraction \"" << value << "\"\n";
                return false;
            }
        } else if (key == "ranked_mutation_enabled") {
            if (!parse_bool(value, config.ranked_mutation_enabled)) {
                std::cerr << "Error: invalid ranked_mutation_enabled \"" << value << "\"\n";
                return false;
            }
        } else if (key == "ranked_mutation_min_multiplier") {
            if (!parse_double(value, config.ranked_mutation_min_multiplier)) {
                std::cerr << "Error: invalid ranked_mutation_min_multiplier \"" << value << "\"\n";
                return false;
            }
        } else if (key == "ranked_mutation_max_multiplier") {
            if (!parse_double(value, config.ranked_mutation_max_multiplier)) {
                std::cerr << "Error: invalid ranked_mutation_max_multiplier \"" << value << "\"\n";
                return false;
            }
        } else if (key == "ranked_mutation_normalize") {
            if (!parse_bool(value, config.ranked_mutation_normalize)) {
                std::cerr << "Error: invalid ranked_mutation_normalize \"" << value << "\"\n";
                return false;
            }
        } else if (key == "balancer_enabled") {
            if (!parse_bool(value, config.balancer_enabled)) {
                std::cerr << "Error: invalid balancer_enabled \"" << value << "\"\n";
                return false;
            }
        } else if (key == "balancer_min_fraction") {
            if (!parse_double(value, config.balancer_min_fraction)) {
                std::cerr << "Error: invalid balancer_min_fraction \"" << value << "\"\n";
                return false;
            }
        } else if (key == "balancer_max_fraction") {
            if (!parse_double(value, config.balancer_max_fraction)) {
                std::cerr << "Error: invalid balancer_max_fraction \"" << value << "\"\n";
                return false;
            }
        } else {
            std::cerr << "Error: unknown config key \"" << key << "\"\n";
            return false;
        }
    }

    return true;
}

bool finalize_config(GAConfig& config, int num_vertices) {
    if (config.population_size < 2) {
        std::cerr << "Error: population_size must be at least 2\n";
        return false;
    }
    if (config.nfe_budget < 1) {
        std::cerr << "Error: nfe_budget must be at least 1\n";
        return false;
    }
    if (config.population_size > config.nfe_budget) {
        std::cerr << "Error: population_size cannot exceed nfe_budget\n";
        return false;
    }
    if (config.tournament_size < 2) {
        std::cerr << "Error: tournament_size must be at least 2\n";
        return false;
    }
    if (config.tournament_size > config.population_size) {
        std::cerr << "Error: tournament_size cannot exceed population_size\n";
        return false;
    }
    if (config.elitism_count < 0 || config.elitism_count >= config.population_size) {
        std::cerr << "Error: elitism_count must be in [0, population_size)\n";
        return false;
    }
    if (config.crossover_rate < 0.0 || config.crossover_rate > 1.0) {
        std::cerr << "Error: crossover_rate must be in [0, 1]\n";
        return false;
    }
    if (config.max_generations < 0) {
        std::cerr << "Error: max_generations must be >= 0\n";
        return false;
    }

    if (config.auto_mutation_rate) {
        config.mutation_rate = 1.0 / num_vertices;
    } else if (config.mutation_rate < 0.0 || config.mutation_rate > 1.0) {
        std::cerr << "Error: mutation_rate must be in [0, 1]\n";
        return false;
    }
    if (config.greedy_by_node_repair_rate < 0.0 || config.greedy_by_node_repair_rate > 1.0) {
        std::cerr << "Error: greedy_by_node_repair_rate must be in [0, 1]\n";
        return false;
    }
    if (config.greedy_by_node_preserve_fraction <= 0.0 ||
        config.greedy_by_node_preserve_fraction > 1.0) {
        std::cerr << "Error: greedy_by_node_preserve_fraction must be in (0, 1]\n";
        return false;
    }
    if (config.ranked_mutation_min_multiplier < 0.0) {
        std::cerr << "Error: ranked_mutation_min_multiplier must be >= 0\n";
        return false;
    }
    if (config.ranked_mutation_max_multiplier <= 0.0) {
        std::cerr << "Error: ranked_mutation_max_multiplier must be > 0\n";
        return false;
    }
    if (config.ranked_mutation_min_multiplier > config.ranked_mutation_max_multiplier) {
        std::cerr << "Error: ranked_mutation_min_multiplier must be <= ranked_mutation_max_multiplier\n";
        return false;
    }
    if (config.balancer_min_fraction <= 0.0 || config.balancer_min_fraction >= 0.5) {
        std::cerr << "Error: balancer_min_fraction must be in (0, 0.5)\n";
        return false;
    }
    if (config.balancer_max_fraction <= 0.5 || config.balancer_max_fraction >= 1.0) {
        std::cerr << "Error: balancer_max_fraction must be in (0.5, 1)\n";
        return false;
    }
    if (config.balancer_min_fraction >= config.balancer_max_fraction) {
        std::cerr << "Error: balancer_min_fraction must be < balancer_max_fraction\n";
        return false;
    }

    return true;
}

bool load_graph(const std::string& path, Graph& graph) {
    std::ifstream input(path);
    if (!input) {
        std::cerr << "Error: could not open \"" << path << "\"\n";
        return false;
    }

    int num_edges = 0;
    if (!(input >> graph.num_vertices >> num_edges)) {
        std::cerr << "Error: invalid header in \"" << path << "\" (expected: N E)\n";
        return false;
    }

    graph.edges.clear();
    graph.edges.reserve(num_edges);

    int u = 0;
    int v = 0;
    int weight = 0;
    while (input >> u >> v >> weight) {
        if (u < 1 || u > graph.num_vertices || v < 1 || v > graph.num_vertices) {
            std::cerr << "Error: vertex index out of range in edge (" << u << ", " << v << ")\n";
            return false;
        }
        graph.edges.push_back({u - 1, v - 1, weight});
    }

    if (static_cast<int>(graph.edges.size()) != num_edges) {
        std::cerr << "Warning: expected " << num_edges << " edges, read "
                  << graph.edges.size() << "\n";
    }

    graph.adjacency.assign(graph.num_vertices, {});
    for (const Edge& edge : graph.edges) {
        graph.adjacency[edge.u].emplace_back(edge.v, edge.weight);
        graph.adjacency[edge.v].emplace_back(edge.u, edge.weight);
    }

    return true;
}

NodeRanking build_node_ranking(const Graph& graph) {
    NodeRanking ranking;
    const int num_vertices = graph.num_vertices;

    std::vector<std::pair<long long, int>> weighted_nodes;
    weighted_nodes.reserve(num_vertices);
    for (int node = 0; node < num_vertices; ++node) {
        long long total_weight = 0;
        for (const auto& [neighbor, weight] : graph.adjacency[node]) {
            total_weight += std::llabs(static_cast<long long>(weight));
        }
        weighted_nodes.emplace_back(total_weight, node);
    }

    std::sort(weighted_nodes.rbegin(), weighted_nodes.rend());
    ranking.order.reserve(num_vertices);
    ranking.position.assign(num_vertices, 0);
    for (int rank = 0; rank < num_vertices; ++rank) {
        const int node = weighted_nodes[rank].second;
        ranking.order.push_back(node);
        ranking.position[node] = rank;
    }

    return ranking;
}

RankedMutationProfile build_ranked_mutation_profile(
    const NodeRanking& ranking,
    int num_vertices,
    double base_rate,
    double min_multiplier,
    double max_multiplier,
    bool normalize) {
    RankedMutationProfile profile;
    profile.per_node_rate.assign(num_vertices, 0.0);

    if (num_vertices <= 1) {
        return profile;
    }

    const double rank_denominator = std::max(num_vertices - 1, 1);
    double multiplier_sum = 0.0;
    int mutable_node_count = 0;

    for (int node = 1; node < num_vertices; ++node) {
        const double rank_fraction = static_cast<double>(ranking.position[node]) / rank_denominator;
        const double multiplier = min_multiplier + rank_fraction * (max_multiplier - min_multiplier);
        profile.per_node_rate[node] = base_rate * multiplier;
        multiplier_sum += multiplier;
        ++mutable_node_count;
    }

    if (normalize && mutable_node_count > 0) {
        const double mean_multiplier = multiplier_sum / mutable_node_count;
        for (int node = 1; node < num_vertices; ++node) {
            profile.per_node_rate[node] /= mean_multiplier;
        }
    }

    profile.min_rate = profile.per_node_rate[1];
    profile.max_rate = profile.per_node_rate[1];
    for (int node = 2; node < num_vertices; ++node) {
        profile.min_rate = std::min(profile.min_rate, profile.per_node_rate[node]);
        profile.max_rate = std::max(profile.max_rate, profile.per_node_rate[node]);
    }

    return profile;
}

GreedyByNodeHeuristic build_greedy_by_node_heuristic(
    const Graph& graph,
    const NodeRanking& ranking,
    double preserve_fraction,
    std::mt19937& rng) {
    GreedyByNodeHeuristic heuristic;
    const int num_vertices = graph.num_vertices;

    heuristic.preserved_node_count = static_cast<int>(std::ceil(num_vertices * preserve_fraction));
    if (heuristic.preserved_node_count > num_vertices) {
        heuristic.preserved_node_count = num_vertices;
    }

    std::vector<int> assignment(num_vertices, -1);
    assignment[0] = 0;

    std::uniform_int_distribution<int> bit_dist(0, 1);

    for (const int node : ranking.order) {
        if (node == 0) {
            continue;
        }

        long long contribution_if_zero = 0;
        long long contribution_if_one = 0;
        bool has_assigned_neighbor = false;

        for (const auto& [neighbor, weight] : graph.adjacency[node]) {
            if (assignment[neighbor] == -1) {
                continue;
            }

            has_assigned_neighbor = true;
            if (assignment[neighbor] == 1) {
                contribution_if_zero += weight;
            } else {
                contribution_if_one += weight;
            }
        }

        if (!has_assigned_neighbor) {
            assignment[node] = bit_dist(rng);
        } else if (contribution_if_zero > contribution_if_one) {
            assignment[node] = 0;
        } else if (contribution_if_one > contribution_if_zero) {
            assignment[node] = 1;
        } else {
            assignment[node] = bit_dist(rng);
        }
    }

    for (int node = 0; node < num_vertices; ++node) {
        if (assignment[node] == -1) {
            assignment[node] = bit_dist(rng);
        }
    }

    assignment[0] = 0;
    heuristic.greedy_solution = std::move(assignment);
    return heuristic;
}

void apply_greedy_by_node_repair(
    Individual& individual,
    const GreedyByNodeHeuristic& heuristic,
    const NodeRanking& ranking) {
    for (int i = 0; i < heuristic.preserved_node_count; ++i) {
        const int node = ranking.order[i];
        individual.genes[node] = heuristic.greedy_solution[node];
    }
    individual.genes[0] = 0;
    individual.evaluated = false;
}

long long evaluate_cut(const Graph& graph, const std::vector<int>& solution, FitnessStats& stats) {
    ++stats.evaluations;

    long long cut_value = 0;
    for (const Edge& edge : graph.edges) {
        if (solution[edge.u] != solution[edge.v]) {
            cut_value += edge.weight;
        }
    }
    return cut_value;
}

void evaluate_individual(const Graph& graph, Individual& individual, FitnessStats& stats) {
    individual.fitness = evaluate_cut(graph, individual.genes, stats);
    individual.evaluated = true;
}

Individual make_random_individual(int num_vertices, std::mt19937& rng) {
    Individual individual;
    individual.genes.assign(num_vertices, 0);
    individual.genes[0] = 0;

    std::uniform_int_distribution<int> bit_dist(0, 1);
    for (int i = 1; i < num_vertices; ++i) {
        individual.genes[i] = bit_dist(rng);
    }

    individual.evaluated = false;
    return individual;
}

int tournament_select(const std::vector<Individual>& population, std::mt19937& rng, int tournament_size) {
    std::uniform_int_distribution<int> index_dist(0, static_cast<int>(population.size()) - 1);

    int best_index = index_dist(rng);
    for (int i = 1; i < tournament_size; ++i) {
        int candidate = index_dist(rng);
        if (population[candidate].fitness > population[best_index].fitness) {
            best_index = candidate;
        }
    }
    return best_index;
}

Individual uniform_crossover(const Individual& parent_a, const Individual& parent_b, std::mt19937& rng) {
    Individual child;
    child.genes = parent_a.genes;
    child.genes[0] = 0;

    std::uniform_int_distribution<int> bit_dist(0, 1);
    for (int i = 1; i < static_cast<int>(child.genes.size()); ++i) {
        if (bit_dist(rng) == 0) {
            child.genes[i] = parent_b.genes[i];
        }
    }

    child.evaluated = false;
    return child;
}

long long compute_flip_gain(const Graph& graph, const std::vector<int>& genes, int node) {
    long long gain = 0;
    for (const auto& [neighbor, weight] : graph.adjacency[node]) {
        if (genes[node] == genes[neighbor]) {
            gain += weight;
        } else {
            gain -= weight;
        }
    }
    return gain;
}

int count_partition_side(const std::vector<int>& genes, int side) {
    int count = 0;
    for (const int gene : genes) {
        if (gene == side) {
            ++count;
        }
    }
    return count;
}

void apply_balancer(Individual& individual, const Graph& graph, const GAConfig& config) {
    const int num_vertices = static_cast<int>(individual.genes.size());
    if (num_vertices <= 1) {
        return;
    }

    individual.genes[0] = 0;

    const int count_ones = count_partition_side(individual.genes, 1);
    const int min_ones = static_cast<int>(std::ceil(config.balancer_min_fraction * num_vertices));
    const int max_ones = static_cast<int>(std::floor(config.balancer_max_fraction * num_vertices));

    if (count_ones >= min_ones && count_ones <= max_ones) {
        return;
    }

    const int larger_side = (count_ones > num_vertices - count_ones) ? 1 : 0;
    int flips_needed = 0;
    if (larger_side == 1) {
        flips_needed = count_ones - max_ones;
    } else {
        flips_needed = min_ones - count_ones;
    }

    if (flips_needed <= 0) {
        return;
    }

    std::vector<std::pair<long long, int>> candidates;
    candidates.reserve(num_vertices);
    for (int node = 1; node < num_vertices; ++node) {
        if (individual.genes[node] == larger_side) {
            candidates.emplace_back(compute_flip_gain(graph, individual.genes, node), node);
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const std::pair<long long, int>& a, const std::pair<long long, int>& b) {
                  if (a.first != b.first) {
                      return a.first > b.first;
                  }
                  return a.second < b.second;
              });

    const int flips_to_apply = std::min(flips_needed, static_cast<int>(candidates.size()));
    for (int i = static_cast<int>(candidates.size()) - flips_to_apply;
         i < static_cast<int>(candidates.size());
         ++i) {
        const int node = candidates[i].second;
        individual.genes[node] = 1 - individual.genes[node];
    }

    individual.genes[0] = 0;
    individual.evaluated = false;
}

void mutate(
    Individual& individual,
    double flat_mutation_rate,
    const std::optional<RankedMutationProfile>& ranked_mutation,
    std::mt19937& rng) {
    individual.genes[0] = 0;

    std::uniform_real_distribution<double> probability_dist(0.0, 1.0);
    for (int i = 1; i < static_cast<int>(individual.genes.size()); ++i) {
        const double mutation_rate = ranked_mutation.has_value()
            ? ranked_mutation->per_node_rate[i]
            : flat_mutation_rate;
        if (probability_dist(rng) < mutation_rate) {
            individual.genes[i] = 1 - individual.genes[i];
        }
    }

    individual.evaluated = false;
}

GAResult run_genetic_algorithm(
    const Graph& graph,
    const GAConfig& config,
    std::mt19937& rng,
    const std::optional<NodeRanking>& node_ranking,
    const std::optional<GreedyByNodeHeuristic>& greedy_by_node,
    const std::optional<RankedMutationProfile>& ranked_mutation) {
    FitnessStats stats;
    GAResult result;

    std::vector<Individual> population;
    population.reserve(config.population_size);

    for (int i = 0; i < config.population_size; ++i) {
        Individual individual = make_random_individual(graph.num_vertices, rng);
        evaluate_individual(graph, individual, stats);
        population.push_back(std::move(individual));
    }

    Individual best = *std::max_element(
        population.begin(),
        population.end(),
        [](const Individual& a, const Individual& b) { return a.fitness < b.fitness; });

    const auto start_time = std::chrono::high_resolution_clock::now();

    int generation = 0;
    while (stats.evaluations < config.nfe_budget) {
        if (config.max_generations > 0 && generation >= config.max_generations) {
            result.stop_reason = "max_generations cap reached";
            break;
        }

        ++generation;

        std::vector<Individual> next_population;
        next_population.reserve(config.population_size);

        std::vector<Individual> ranked = population;
        std::sort(ranked.begin(), ranked.end(),
                  [](const Individual& a, const Individual& b) { return a.fitness > b.fitness; });

        for (int i = 0; i < config.elitism_count && i < static_cast<int>(ranked.size()); ++i) {
            next_population.push_back(ranked[i]);
        }

        std::uniform_real_distribution<double> crossover_dist(0.0, 1.0);
        std::uniform_real_distribution<double> repair_dist(0.0, 1.0);

        while (static_cast<int>(next_population.size()) < config.population_size &&
               stats.evaluations < config.nfe_budget) {
            const int parent_a_index = tournament_select(population, rng, config.tournament_size);
            const int parent_b_index = tournament_select(population, rng, config.tournament_size);

            Individual child = population[parent_a_index];
            if (crossover_dist(rng) < config.crossover_rate) {
                child = uniform_crossover(population[parent_a_index], population[parent_b_index], rng);
            }

            if (greedy_by_node.has_value() && node_ranking.has_value() &&
                repair_dist(rng) < config.greedy_by_node_repair_rate) {
                apply_greedy_by_node_repair(child, greedy_by_node.value(), node_ranking.value());
            }

            mutate(child, config.mutation_rate, ranked_mutation, rng);

            if (config.balancer_enabled) {
                apply_balancer(child, graph, config);
            }

            evaluate_individual(graph, child, stats);
            next_population.push_back(std::move(child));
        }

        population = std::move(next_population);

        const Individual& generation_best = *std::max_element(
            population.begin(),
            population.end(),
            [](const Individual& a, const Individual& b) { return a.fitness < b.fitness; });

        if (generation_best.fitness > best.fitness) {
            best = generation_best;
        }

        result.generations_run = generation;

        if (stats.evaluations >= config.nfe_budget) {
            result.stop_reason = "NFE budget reached";
            break;
        }
    }

    if (result.stop_reason.empty()) {
        result.stop_reason = "NFE budget reached";
    }

    const auto end_time = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    result.best_solution = best.genes;
    result.best_fitness = best.fitness;
    result.fitness_evaluations = stats.evaluations;
    return result;
}

void print_config(
    const GAConfig& config,
    const std::string& config_path,
    bool auto_mutation,
    const std::optional<GreedyByNodeHeuristic>& greedy_by_node,
    const std::optional<RankedMutationProfile>& ranked_mutation) {
    std::cout << "Config file: " << config_path << "\n";
    std::cout << "Population: " << config.population_size << "\n";
    std::cout << "Stop condition: NFE budget = " << config.nfe_budget << "\n";
    if (config.max_generations > 0) {
        std::cout << "Safety cap: max_generations = " << config.max_generations << "\n";
    }
    std::cout << "Crossover rate: " << config.crossover_rate
              << " (uniform), Tournament size: " << config.tournament_size << "\n";
    std::cout << "Mutation rate: " << config.mutation_rate;
    if (auto_mutation) {
        std::cout << " (auto: 1/N)";
    }
    std::cout << "\nElitism: " << config.elitism_count << "\n";
    if (config.random_seed != 0) {
        std::cout << "Random seed: " << config.random_seed << "\n";
    }
    std::cout << "Greedy-by-node repair: "
              << (config.greedy_by_node_enabled ? "enabled" : "disabled") << "\n";
    if (config.greedy_by_node_enabled && greedy_by_node.has_value()) {
        std::cout << "  Repair rate: " << config.greedy_by_node_repair_rate
                  << " of offspring\n";
        std::cout << "  Preserved nodes: " << greedy_by_node->preserved_node_count
                  << " / " << greedy_by_node->greedy_solution.size()
                  << " (top " << (config.greedy_by_node_preserve_fraction * 100.0)
                  << "%, rounded up)\n";
    }
    std::cout << "Ranked mutation: "
              << (config.ranked_mutation_enabled ? "enabled" : "disabled") << "\n";
    if (config.ranked_mutation_enabled && ranked_mutation.has_value()) {
        std::cout << "  Spread: linear, rank 0 (top weight) -> rank N-1 (bottom)\n";
        std::cout << "  Multipliers: " << config.ranked_mutation_min_multiplier
                  << " (top) to " << config.ranked_mutation_max_multiplier
                  << " (bottom)\n";
        std::cout << "  Normalize: "
                  << (config.ranked_mutation_normalize ? "yes (~1 expected flip/offspring)"
                                                       : "no")
                  << "\n";
        std::cout << "  Per-node rate range: " << ranked_mutation->min_rate
                  << " .. " << ranked_mutation->max_rate << "\n";
    }
    std::cout << "Balancer: " << (config.balancer_enabled ? "enabled" : "disabled") << "\n";
    if (config.balancer_enabled) {
        std::cout << "  Target partition ratio: "
                  << (config.balancer_min_fraction * 100.0) << "% .. "
                  << (config.balancer_max_fraction * 100.0) << "% for partition S\n";
        std::cout << "  Strategy: flip lowest-gain nodes on the larger side (vertex 0 fixed)\n";
    }
}

void print_result(const GAResult& result, long long nfe_budget) {
    std::cout << "\nBest cut value: " << result.best_fitness << "\n";
    std::cout << "Fitness evaluations: " << result.fitness_evaluations
              << " / " << nfe_budget << "\n";
    std::cout << "Generations: " << result.generations_run << "\n";
    std::cout << "Stop reason: " << result.stop_reason << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Elapsed time (ms): " << result.elapsed_ms << "\n";
    std::cout << "Solution (0/1 partition, vertex 0 fixed to 0):\n";

    for (size_t i = 0; i < result.best_solution.size(); ++i) {
        std::cout << result.best_solution[i];
        if (i + 1 < result.best_solution.size()) {
            std::cout << ' ';
        }
    }
    std::cout << "\n";
}

void print_result_json(
    const GAResult& result,
    const GAConfig& config,
    const std::string& instance_path,
    const std::string& config_path) {
    std::cout << "{";
    std::cout << "\"instance\":\"" << instance_path << "\"";
    std::cout << ",\"config\":\"" << config_path << "\"";
    std::cout << ",\"heuristics\":{";
    std::cout << "\"greedy_by_node\":" << (config.greedy_by_node_enabled ? "true" : "false");
    std::cout << ",\"ranked_mutation\":" << (config.ranked_mutation_enabled ? "true" : "false");
    std::cout << ",\"balancer\":" << (config.balancer_enabled ? "true" : "false");
    std::cout << "}";
    std::cout << ",\"cut_value\":" << result.best_fitness;
    std::cout << ",\"nfe\":" << result.fitness_evaluations;
    std::cout << ",\"nfe_budget\":" << config.nfe_budget;
    std::cout << ",\"generations\":" << result.generations_run;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << ",\"elapsed_ms\":" << result.elapsed_ms;
    std::cout << ",\"stop_reason\":\"" << result.stop_reason << "\"";
    std::cout << ",\"solution\":[";
    for (size_t i = 0; i < result.best_solution.size(); ++i) {
        std::cout << result.best_solution[i];
        if (i + 1 < result.best_solution.size()) {
            std::cout << ',';
        }
    }
    std::cout << "]";
    std::cout << "}\n";
}

int main(int argc, char* argv[]) {
    bool json_mode = false;
    std::string instance_path = "Instances/t2g10_5555.txt";
    std::string config_path = "ga_config.txt";

    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--json") {
            json_mode = true;
        } else if (arg[0] != '-') {
            if (positional == 0) {
                instance_path = arg;
            } else if (positional == 1) {
                config_path = arg;
            }
            ++positional;
        }
    }

    Graph graph;
    if (!load_graph(instance_path, graph)) {
        return 1;
    }

    GAConfig config;
    if (!load_config(config_path, config)) {
        return 1;
    }

    const bool auto_mutation = config.auto_mutation_rate;
    if (!finalize_config(config, graph.num_vertices)) {
        return 1;
    }

    std::mt19937 rng;
    if (config.random_seed != 0) {
        rng.seed(config.random_seed);
    } else {
        std::random_device random_device;
        rng.seed(random_device());
    }

    std::optional<NodeRanking> node_ranking;
    if (config.greedy_by_node_enabled || config.ranked_mutation_enabled) {
        node_ranking = build_node_ranking(graph);
    }

    std::optional<GreedyByNodeHeuristic> greedy_by_node;
    if (config.greedy_by_node_enabled && node_ranking.has_value()) {
        greedy_by_node = build_greedy_by_node_heuristic(
            graph,
            node_ranking.value(),
            config.greedy_by_node_preserve_fraction,
            rng);
    }

    std::optional<RankedMutationProfile> ranked_mutation;
    if (config.ranked_mutation_enabled && node_ranking.has_value()) {
        ranked_mutation = build_ranked_mutation_profile(
            node_ranking.value(),
            graph.num_vertices,
            config.mutation_rate,
            config.ranked_mutation_min_multiplier,
            config.ranked_mutation_max_multiplier,
            config.ranked_mutation_normalize);
    }

    if (!json_mode) {
        std::cout << "Baseline Genetic Algorithm (Max-Cut)\n";
        std::cout << "Instance: " << instance_path << "\n";
        std::cout << "Vertices: " << graph.num_vertices << ", Edges: " << graph.edges.size() << "\n";
        print_config(config, config_path, auto_mutation, greedy_by_node, ranked_mutation);
        std::cout << "\n";
    }

    const GAResult result = run_genetic_algorithm(
        graph, config, rng, node_ranking, greedy_by_node, ranked_mutation);

    if (json_mode) {
        print_result_json(result, config, instance_path, config_path);
    } else {
        print_result(result, config.nfe_budget);
    }
    return 0;
}
