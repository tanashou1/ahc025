use std::io::{self, Write};

const UNKNOWN: i8 = 2;
const MEAN_WEIGHT: f64 = 100_000.0;
const EPS: f64 = 1e-9;

fn logging_enabled() -> bool {
    std::env::var_os("AHC025_LOG").is_some()
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn estimated_imbalance(group_sum: &[f64]) -> f64 {
    if group_sum.is_empty() {
        return 0.0;
    }
    let d = group_sum.len() as f64;
    let total: f64 = group_sum.iter().sum();
    let sum_sq: f64 = group_sum.iter().map(|&x| x * x).sum();
    ((sum_sq * d - total * total).max(0.0)).sqrt() / d
}

fn format_group_sums(group_sum: &[f64]) -> String {
    group_sum
        .iter()
        .map(|sum| format!("{sum:.1}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_group_sizes(groups: &[Vec<usize>]) -> String {
    groups
        .iter()
        .map(|group| group.len().to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn ceil_log2(x: usize) -> usize {
    let mut r = 0;
    let mut v = 1;
    while v < x {
        v <<= 1;
        r += 1;
    }
    r
}

fn full_sort_cost(n: usize) -> usize {
    (1..n).map(|len| ceil_log2(len + 1)).sum()
}

struct XorShift64 {
    x: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let x = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
        Self { x }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.x;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.x = x;
        x
    }

    fn gen_range(&mut self, upper: usize) -> usize {
        (self.next_u64() % upper as u64) as usize
    }
}

struct Judge {
    n: usize,
    d: usize,
    q: usize,
    used: usize,
    item_cmp: Vec<Vec<i8>>,
}

impl Judge {
    fn new() -> Self {
        let mut first = String::new();
        io::stdin().read_line(&mut first).unwrap();
        let vals: Vec<usize> = first
            .split_whitespace()
            .map(|s| s.parse::<usize>().unwrap())
            .collect();
        let n = vals[0];
        let d = vals[1];
        let q = vals[2];
        let mut item_cmp = vec![vec![UNKNOWN; n]; n];
        for (i, row) in item_cmp.iter_mut().enumerate() {
            row[i] = 0;
        }
        Self {
            n,
            d,
            q,
            used: 0,
            item_cmp,
        }
    }

    fn remaining(&self) -> usize {
        self.q - self.used
    }

    fn compare_sets(&mut self, left: &[usize], right: &[usize]) -> i8 {
        assert!(!left.is_empty());
        assert!(!right.is_empty());
        assert!(self.used < self.q);

        print!("{} {}", left.len(), right.len());
        for &x in left {
            print!(" {}", x);
        }
        for &x in right {
            print!(" {}", x);
        }
        println!();
        io::stdout().flush().unwrap();

        let mut response = String::new();
        io::stdin().read_line(&mut response).unwrap();
        self.used += 1;
        match response.trim() {
            "<" => -1,
            ">" => 1,
            "=" => 0,
            other => panic!("unexpected judge response: {other}"),
        }
    }

    fn compare_items_raw(&mut self, a: usize, b: usize) -> i8 {
        if self.item_cmp[a][b] != UNKNOWN {
            return self.item_cmp[a][b];
        }
        let res = self.compare_sets(&[a], &[b]);
        self.item_cmp[a][b] = res;
        self.item_cmp[b][a] = -res;
        res
    }

    fn item_better(&mut self, a: usize, b: usize) -> bool {
        let res = self.compare_items_raw(a, b);
        if res != 0 {
            res > 0
        } else {
            a < b
        }
    }

    fn filler_query(&mut self) {
        let _ = self.compare_sets(&[0], &[1]);
    }

    fn fill_rest(&mut self) {
        while self.used < self.q {
            self.filler_query();
        }
    }
}

struct ItemHeap {
    heap: Vec<usize>,
}

impl ItemHeap {
    fn new() -> Self {
        Self { heap: Vec::new() }
    }

    fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    fn len(&self) -> usize {
        self.heap.len()
    }

    fn top(&self) -> usize {
        self.heap[0]
    }

    fn push(&mut self, x: usize, judge: &mut Judge) {
        self.heap.push(x);
        let mut i = self.heap.len() - 1;
        while i > 0 {
            let p = (i - 1) / 2;
            if !judge.item_better(self.heap[i], self.heap[p]) {
                break;
            }
            self.heap.swap(i, p);
            i = p;
        }
    }

    fn pop(&mut self, judge: &mut Judge) -> usize {
        let ret = self.heap[0];
        let last = self.heap.pop().unwrap();
        if !self.heap.is_empty() {
            self.heap[0] = last;
            let mut i = 0;
            loop {
                let l = i * 2 + 1;
                let r = l + 1;
                if l >= self.heap.len() {
                    break;
                }
                let mut best = l;
                if r < self.heap.len() && judge.item_better(self.heap[r], self.heap[l]) {
                    best = r;
                }
                if !judge.item_better(self.heap[best], self.heap[i]) {
                    break;
                }
                self.heap.swap(i, best);
                i = best;
            }
        }
        ret
    }
}

fn heap_pop_cost(heap_size: usize) -> usize {
    if heap_size <= 1 {
        0
    } else {
        env_usize("AHC025_HEAP_POP_COEF").unwrap_or(2) * ceil_log2(heap_size)
            + env_usize("AHC025_HEAP_POP_EXTRA").unwrap_or(0)
    }
}

fn heap_push_cost(new_size: usize) -> usize {
    ceil_log2(new_size.max(1)) + env_usize("AHC025_HEAP_PUSH_EXTRA").unwrap_or(0)
}

fn extraction_cost_upper_bound(heap_size: usize, child_count: usize) -> usize {
    let mut cost = heap_pop_cost(heap_size);
    let mut cur_size = heap_size.saturating_sub(1);
    for _ in 0..child_count {
        cur_size += 1;
        cost += heap_push_cost(cur_size);
    }
    cost
}

struct TournamentData {
    champion: usize,
    children: Vec<Vec<usize>>,
}

fn shuffle<T>(values: &mut [T], seed: u64) {
    let mut rng = XorShift64::new(seed);
    for i in (1..values.len()).rev() {
        let j = rng.gen_range(i + 1);
        values.swap(i, j);
    }
}

fn build_exact_order_by_insertion(judge: &mut Judge, reserve_queries: usize) -> Option<Vec<usize>> {
    let need = full_sort_cost(judge.n);
    if judge.remaining() < reserve_queries + need {
        return None;
    }

    let seed = (judge.n as u64) * 1_000_003 + (judge.d as u64) * 10_007 + judge.q as u64 + 0x1234_5678_9abc_def0;
    let mut items: Vec<usize> = (0..judge.n).collect();
    shuffle(&mut items, seed);

    let mut sorted = vec![items[0]];
    for &item in &items[1..] {
        let mut lo = 0usize;
        let mut hi = sorted.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if judge.item_better(item, sorted[mid]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        sorted.insert(lo, item);
    }
    Some(sorted)
}

fn build_tournament(judge: &mut Judge) -> TournamentData {
    let mut items: Vec<usize> = (0..judge.n).collect();
    let seed = (judge.n as u64) * 1_000_003 + (judge.d as u64) * 10_007 + judge.q as u64 + 0x9e37_79b9_7f4a_7c15;
    shuffle(&mut items, seed);

    let mut children = vec![Vec::new(); judge.n];
    let mut current = items;
    while current.len() > 1 {
        let mut next = Vec::with_capacity((current.len() + 1) / 2);
        let mut i = 0;
        while i + 1 < current.len() {
            let a = current[i];
            let b = current[i + 1];
            if judge.item_better(a, b) {
                children[a].push(b);
                next.push(a);
            } else {
                children[b].push(a);
                next.push(b);
            }
            i += 2;
        }
        if i < current.len() {
            next.push(current[i]);
        }
        current = next;
    }

    TournamentData {
        champion: current[0],
        children,
    }
}

fn build_exact_prefix(judge: &mut Judge, tournament: &TournamentData, reserve_queries: usize) -> Vec<usize> {
    let mut exact_order = vec![tournament.champion];
    let mut frontier = ItemHeap::new();
    for &child in &tournament.children[tournament.champion] {
        frontier.push(child, judge);
    }

    while !frontier.is_empty() {
        let candidate = frontier.top();
        let need = extraction_cost_upper_bound(frontier.len(), tournament.children[candidate].len());
        let effective_reserve = effective_prefix_reserve(reserve_queries, exact_order.len(), judge.d);
        if judge.remaining() < effective_reserve + need {
            break;
        }
        let x = frontier.pop(judge);
        exact_order.push(x);
        for &child in &tournament.children[x] {
            frontier.push(child, judge);
        }
    }

    exact_order
}

#[derive(Clone, Copy)]
struct Candidate {
    parent_pos: usize,
    wins: usize,
    item: usize,
}

fn build_full_order(n: usize, exact_order: &[usize], children: &[Vec<usize>]) -> Vec<usize> {
    let mut used = vec![false; n];
    let mut order = exact_order.to_vec();
    for &x in exact_order {
        used[x] = true;
    }

    let mut pool: Vec<Candidate> = Vec::new();

    for (parent_pos, &u) in exact_order.iter().enumerate() {
        let mut ch = children[u].clone();
        ch.sort_by(|&a, &b| children[b].len().cmp(&children[a].len()).then_with(|| a.cmp(&b)));
        for v in ch {
            if !used[v] {
                pool.push(Candidate {
                    parent_pos,
                    wins: children[v].len(),
                    item: v,
                });
            }
        }
    }

    while order.len() < n && !pool.is_empty() {
        let mut best_idx = 0usize;
        for i in 1..pool.len() {
            let cur = pool[i];
            let best = pool[best_idx];
            let better = if cur.parent_pos != best.parent_pos {
                cur.parent_pos < best.parent_pos
            } else if cur.wins != best.wins {
                cur.wins > best.wins
            } else {
                cur.item < best.item
            };
            if better {
                best_idx = i;
            }
        }

        let picked = pool.swap_remove(best_idx);
        if used[picked.item] {
            continue;
        }
        used[picked.item] = true;
        order.push(picked.item);

        let mut ch = children[picked.item].clone();
        ch.sort_by(|&a, &b| children[b].len().cmp(&children[a].len()).then_with(|| a.cmp(&b)));
        let parent_pos = order.len() - 1;
        for v in ch {
            if !used[v] {
                pool.push(Candidate {
                    parent_pos,
                    wins: children[v].len(),
                    item: v,
                });
            }
        }
    }

    for (i, &seen) in used.iter().enumerate() {
        if !seen {
            order.push(i);
        }
    }
    order
}

fn build_rank_weights(n: usize, d: usize) -> Vec<f64> {
    let mut harmonic = vec![0.0; n + 1];
    for i in 1..=n {
        harmonic[i] = harmonic[i - 1] + 1.0 / i as f64;
    }

    let cap = MEAN_WEIGHT * n as f64 / d as f64;
    (0..n)
        .map(|rank| (MEAN_WEIGHT * (harmonic[n] - harmonic[rank])).min(cap))
        .collect()
}

fn compute_reserve_queries(n: usize, d: usize, q: usize) -> usize {
    let low_q_ratio_num = env_usize("AHC025_LOW_Q_RATIO_NUM").unwrap_or(8);
    let low_q_ratio_den = env_usize("AHC025_LOW_Q_RATIO_DEN").unwrap_or(1).max(1);
    let default_q_div = env_usize("AHC025_RESERVE_Q_DIV").unwrap_or(4).max(1);
    let low_q_div = env_usize("AHC025_LOW_Q_DIV").unwrap_or(8).max(1);
    let q_div = if q * low_q_ratio_den < n * low_q_ratio_num {
        low_q_div
    } else {
        default_q_div
    };
    let d_coef = env_usize("AHC025_RESERVE_D_COEF").unwrap_or(8);
    let n_div = env_usize("AHC025_RESERVE_N_DIV").unwrap_or(0);
    let offset = env_usize("AHC025_RESERVE_OFFSET").unwrap_or(24);
    let mut reserve = (q / q_div).min(d_coef * d + offset);
    if n_div > 0 {
        reserve = reserve.max(n / n_div);
    }
    reserve
}

fn effective_prefix_reserve(reserve_queries: usize, exact_prefix_len: usize, d: usize) -> usize {
    let target_mul_d = env_usize("AHC025_PREFIX_TARGET_MUL_D").unwrap_or(2);
    let reserve_div = env_usize("AHC025_PREFIX_RESERVE_DIV").unwrap_or(5).max(1);
    if target_mul_d > 0 && exact_prefix_len < target_mul_d * d {
        reserve_queries / reserve_div
    } else {
        reserve_queries
    }
}

struct AssignmentState {
    group_of: Vec<usize>,
    groups: Vec<Vec<usize>>,
    group_sum: Vec<f64>,
    item_weight: Vec<f64>,
}

fn build_initial_assignment(n: usize, d: usize, order: &[usize], rank_weight: &[f64]) -> AssignmentState {
    let mut state = AssignmentState {
        group_of: vec![0; n],
        groups: vec![Vec::new(); d],
        group_sum: vec![0.0; d],
        item_weight: vec![0.0; n],
    };

    for (pos, &item) in order.iter().enumerate() {
        state.item_weight[item] = rank_weight[pos];
    }

    for &item in order {
        let mut best = 0usize;
        for g in 1..d {
            let better = if state.group_sum[g] + EPS < state.group_sum[best] {
                true
            } else if (state.group_sum[g] - state.group_sum[best]).abs() <= EPS {
                if state.groups[g].len() != state.groups[best].len() {
                    state.groups[g].len() < state.groups[best].len()
                } else {
                    g < best
                }
            } else {
                false
            };
            if better {
                best = g;
            }
        }
        state.group_of[item] = best;
        state.groups[best].push(item);
        state.group_sum[best] += state.item_weight[item];
    }

    state
}

fn heaviest_movable_group(state: &AssignmentState) -> Option<usize> {
    let mut best: Option<usize> = None;
    for g in 0..state.groups.len() {
        if state.groups[g].len() <= 1 {
            continue;
        }
        match best {
            None => best = Some(g),
            Some(b) => {
                let better = if state.group_sum[g] > state.group_sum[b] + EPS {
                    true
                } else if (state.group_sum[g] - state.group_sum[b]).abs() <= EPS {
                    if state.groups[g].len() != state.groups[b].len() {
                        state.groups[g].len() > state.groups[b].len()
                    } else {
                        g < b
                    }
                } else {
                    false
                };
                if better {
                    best = Some(g);
                }
            }
        }
    }
    best
}

fn lightest_group_excluding(state: &AssignmentState, exclude: usize) -> usize {
    let mut best = if exclude == 0 { 1 } else { 0 };
    for g in 0..state.groups.len() {
        if g == exclude {
            continue;
        }
        let better = if state.group_sum[g] + EPS < state.group_sum[best] {
            true
        } else if (state.group_sum[g] - state.group_sum[best]).abs() <= EPS {
            if state.groups[g].len() != state.groups[best].len() {
                state.groups[g].len() < state.groups[best].len()
            } else {
                g < best
            }
        } else {
            false
        };
        if better {
            best = g;
        }
    }
    best
}

fn build_move_candidates(heavy_group: &[usize], item_weight: &[f64], target: f64) -> Vec<usize> {
    let mut items = heavy_group.to_vec();
    items.sort_by(|&a, &b| {
        item_weight[a]
            .partial_cmp(&item_weight[b])
            .unwrap()
            .then_with(|| a.cmp(&b))
    });

    let mut idx = items.len();
    for (i, &item) in items.iter().enumerate() {
        if item_weight[item] + EPS >= target {
            idx = i;
            break;
        }
    }

    let mut candidates = Vec::new();
    let add_index = |i: isize, candidates: &mut Vec<usize>| {
        if i < 0 || i as usize >= items.len() {
            return;
        }
        let x = items[i as usize];
        if !candidates.contains(&x) {
            candidates.push(x);
        }
    };

    add_index(idx as isize, &mut candidates);
    add_index(idx as isize - 1, &mut candidates);
    add_index(idx as isize + 1, &mut candidates);
    add_index(0, &mut candidates);
    add_index(items.len() as isize - 1, &mut candidates);

    candidates.sort_by(|&a, &b| {
        (item_weight[a] - target)
            .abs()
            .partial_cmp(&(item_weight[b] - target).abs())
            .unwrap()
            .then_with(|| a.cmp(&b))
    });
    candidates
}

fn build_ranked_move_candidates(heavy_group: &[usize], item_weight: &[f64], target: f64) -> Vec<usize> {
    let mut candidates = heavy_group.to_vec();
    candidates.sort_by(|&a, &b| {
        (item_weight[a] - target)
            .abs()
            .partial_cmp(&(item_weight[b] - target).abs())
            .unwrap()
            .then_with(|| item_weight[b].partial_cmp(&item_weight[a]).unwrap())
            .then_with(|| a.cmp(&b))
    });
    candidates
}

fn build_swap_candidates(
    heavy_group: &[usize],
    light_group: &[usize],
    item_weight: &[f64],
    diff: f64,
) -> Vec<(usize, usize)> {
    let heavy_candidates = build_ranked_move_candidates(heavy_group, item_weight, diff / 2.0);
    let mut candidates = Vec::new();
    let heavy_take = env_usize("AHC025_SWAP_HEAVY_TAKE").unwrap_or(5);
    let light_take = env_usize("AHC025_SWAP_LIGHT_TAKE").unwrap_or(5);
    for &heavy_item in heavy_candidates.iter().take(heavy_take) {
        let desired_light = (item_weight[heavy_item] - diff / 2.0).max(0.0);
        let mut light_candidates = light_group.to_vec();
        light_candidates.sort_by(|&a, &b| {
            (item_weight[a] - desired_light)
                .abs()
                .partial_cmp(&(item_weight[b] - desired_light).abs())
                .unwrap()
                .then_with(|| item_weight[a].partial_cmp(&item_weight[b]).unwrap())
                .then_with(|| a.cmp(&b))
        });
        for &light_item in light_candidates.iter().take(light_take) {
            let pair = (heavy_item, light_item);
            if !candidates.contains(&pair) {
                candidates.push(pair);
            }
        }
    }
    candidates
}

fn apply_move(item: usize, from: usize, to: usize, state: &mut AssignmentState) {
    let pos = state.groups[from].iter().position(|&x| x == item).unwrap();
    state.groups[from].swap_remove(pos);
    state.groups[to].push(item);
    state.group_of[item] = to;
    state.group_sum[from] -= state.item_weight[item];
    state.group_sum[to] += state.item_weight[item];
}

fn replace_item(group: &[usize], remove: usize, add: usize) -> Vec<usize> {
    group.iter()
        .copied()
        .map(|item| if item == remove { add } else { item })
        .collect()
}

fn apply_swap(heavy_item: usize, hi: usize, light_item: usize, lo: usize, state: &mut AssignmentState) {
    let hi_pos = state.groups[hi].iter().position(|&x| x == heavy_item).unwrap();
    let lo_pos = state.groups[lo].iter().position(|&x| x == light_item).unwrap();
    state.groups[hi][hi_pos] = light_item;
    state.groups[lo][lo_pos] = heavy_item;
    state.group_of[heavy_item] = lo;
    state.group_of[light_item] = hi;
    state.group_sum[hi] += state.item_weight[light_item] - state.item_weight[heavy_item];
    state.group_sum[lo] += state.item_weight[heavy_item] - state.item_weight[light_item];
}

#[derive(Default)]
struct RefineStats {
    iterations: usize,
    balance_queries: usize,
    candidate_queries: usize,
    equal_group_comparisons: usize,
    accepted_moves: usize,
    rejected_moves: usize,
    no_move_rounds: usize,
}

#[derive(Default)]
struct FinalQueryStats {
    pair_queries: usize,
    candidate_queries: usize,
    accepted_moves: usize,
    exhausted_pairs: usize,
}

#[derive(Default)]
struct SwapStats {
    pair_queries: usize,
    swap_queries: usize,
    accepted_swaps: usize,
    stalled_rounds: usize,
}

fn refine_assignment(judge: &mut Judge, state: &mut AssignmentState) -> RefineStats {
    let mut stats = RefineStats::default();
    if judge.remaining() <= 2 {
        return stats;
    }

    let mut stagnation = 0usize;
    while judge.remaining() > 0 && stagnation < 2 * judge.d {
        stats.iterations += 1;
        let Some(mut hi) = heaviest_movable_group(state) else {
            break;
        };
        let mut lo = lightest_group_excluding(state, hi);

        if judge.remaining() < 2 {
            break;
        }

        stats.balance_queries += 1;
        let cmp = judge.compare_sets(&state.groups[hi], &state.groups[lo]);
        if cmp == 0 {
            stats.equal_group_comparisons += 1;
            stagnation += 1;
            continue;
        }
        if cmp < 0 {
            std::mem::swap(&mut hi, &mut lo);
            if state.groups[hi].len() <= 1 {
                stagnation += 1;
                continue;
            }
        }

        let target = (state.group_sum[hi] - state.group_sum[lo]).abs() / 2.0;
        let candidates = build_move_candidates(&state.groups[hi], &state.item_weight, target);

        let mut moved = false;
        for item in candidates {
            if judge.remaining() == 0 || state.groups[hi].len() <= 1 {
                break;
            }

            let left: Vec<usize> = state.groups[hi]
                .iter()
                .copied()
                .filter(|&x| x != item)
                .collect();
            if left.is_empty() {
                continue;
            }

            let mut right = state.groups[lo].clone();
            right.push(item);
            stats.candidate_queries += 1;
            let move_cmp = judge.compare_sets(&left, &right);
            if move_cmp >= 0 {
                apply_move(item, hi, lo, state);
                stats.accepted_moves += 1;
                moved = true;
                break;
            }
            stats.rejected_moves += 1;
        }

        if moved {
            stagnation = 0;
        } else {
            stats.no_move_rounds += 1;
            stagnation += 1;
        }
    }
    stats
}

fn use_remaining_queries(judge: &mut Judge, state: &mut AssignmentState) -> FinalQueryStats {
    let mut stats = FinalQueryStats::default();
    while judge.remaining() >= 2 {
        let mut pairs = Vec::new();
        for hi in 0..judge.d {
            if state.groups[hi].len() <= 1 {
                continue;
            }
            for lo in 0..judge.d {
                if hi == lo {
                    continue;
                }
                if state.group_sum[hi] <= state.group_sum[lo] + EPS {
                    continue;
                }
                pairs.push((hi, lo));
            }
        }
        if pairs.is_empty() {
            break;
        }
        pairs.sort_by(|&(ahi, alo), &(bhi, blo)| {
            let adiff = state.group_sum[ahi] - state.group_sum[alo];
            let bdiff = state.group_sum[bhi] - state.group_sum[blo];
            bdiff
                .partial_cmp(&adiff)
                .unwrap()
                .then_with(|| ahi.cmp(&bhi))
                .then_with(|| alo.cmp(&blo))
        });

        let mut moved = false;
        for &(mut hi, mut lo) in &pairs {
            if judge.remaining() < 2 {
                break;
            }

            stats.pair_queries += 1;
            let cmp = judge.compare_sets(&state.groups[hi], &state.groups[lo]);
            if cmp == 0 {
                stats.exhausted_pairs += 1;
                continue;
            }
            if cmp < 0 {
                std::mem::swap(&mut hi, &mut lo);
            }
            if state.groups[hi].len() <= 1 {
                stats.exhausted_pairs += 1;
                continue;
            }

            let target = (state.group_sum[hi] - state.group_sum[lo]).abs() / 2.0;
            let candidates = build_ranked_move_candidates(&state.groups[hi], &state.item_weight, target);
            let mut pair_moved = false;
            for item in candidates {
                if judge.remaining() == 0 || state.groups[hi].len() <= 1 {
                    break;
                }

                let left: Vec<usize> = state.groups[hi]
                    .iter()
                    .copied()
                    .filter(|&x| x != item)
                    .collect();
                if left.is_empty() {
                    continue;
                }

                let mut right = state.groups[lo].clone();
                right.push(item);
                stats.candidate_queries += 1;
                let move_cmp = judge.compare_sets(&left, &right);
                if move_cmp >= 0 {
                    apply_move(item, hi, lo, state);
                    stats.accepted_moves += 1;
                    moved = true;
                    pair_moved = true;
                    break;
                }
            }
            if pair_moved {
                break;
            }
            stats.exhausted_pairs += 1;
        }

        if !moved {
            break;
        }
    }
    stats
}

fn insertion_endgame_swaps(judge: &mut Judge, state: &mut AssignmentState) -> SwapStats {
    let mut stats = SwapStats::default();
    let max_stalled = env_usize("AHC025_SWAP_STALLED_FACTOR").unwrap_or(2) * state.groups.len().max(1);
    let pair_take = env_usize("AHC025_SWAP_PAIR_TAKE").unwrap_or(8);
    while judge.remaining() >= 2 && stats.stalled_rounds < max_stalled {
        let Some(mut hi) = heaviest_movable_group(state) else {
            break;
        };
        let mut lo = lightest_group_excluding(state, hi);

        stats.pair_queries += 1;
        let cmp = judge.compare_sets(&state.groups[hi], &state.groups[lo]);
        if cmp == 0 {
            stats.stalled_rounds += 1;
            continue;
        }
        if cmp < 0 {
            std::mem::swap(&mut hi, &mut lo);
        }
        if state.groups[hi].is_empty() || state.groups[lo].is_empty() {
            stats.stalled_rounds += 1;
            continue;
        }

        let diff = (state.group_sum[hi] - state.group_sum[lo]).abs();
        let candidates = build_swap_candidates(&state.groups[hi], &state.groups[lo], &state.item_weight, diff);
        let mut swapped = false;
        for (heavy_item, light_item) in candidates.into_iter().take(pair_take) {
            if judge.remaining() == 0 {
                break;
            }
            let left = replace_item(&state.groups[hi], heavy_item, light_item);
            let right = replace_item(&state.groups[lo], light_item, heavy_item);
            stats.swap_queries += 1;
            let swap_cmp = judge.compare_sets(&left, &right);
            if swap_cmp >= 0 {
                apply_swap(heavy_item, hi, light_item, lo, state);
                stats.accepted_swaps += 1;
                swapped = true;
                break;
            }
        }

        if swapped {
            stats.stalled_rounds = 0;
        } else {
            stats.stalled_rounds += 1;
        }
    }
    stats
}

fn main() {
    let log_enabled = logging_enabled();
    let mut judge = Judge::new();

    let reserve_queries = compute_reserve_queries(judge.n, judge.d, judge.q);
    let (ordering_mode, exact_prefix_len, full_order) =
        if let Some(order) = build_exact_order_by_insertion(&mut judge, reserve_queries) {
            ("insertion", judge.n, order)
        } else {
            let tournament = build_tournament(&mut judge);
            let exact_prefix = build_exact_prefix(&mut judge, &tournament, reserve_queries);
            let exact_prefix_len = exact_prefix.len();
            let full_order = build_full_order(judge.n, &exact_prefix, &tournament.children);
            ("tournament", exact_prefix_len, full_order)
        };
    let queries_after_ordering = judge.used;
    if log_enabled {
        eprintln!(
            "[order] mode={} exact_prefix_len={} queries_used={}/{} reserve_queries={} full_order_len={}",
            ordering_mode,
            exact_prefix_len,
            queries_after_ordering,
            judge.q,
            reserve_queries,
            full_order.len()
        );
    }
    let rank_weight = build_rank_weights(judge.n, judge.d);

    let mut state = build_initial_assignment(judge.n, judge.d, &full_order, &rank_weight);
    let initial_imbalance = estimated_imbalance(&state.group_sum);
    if log_enabled {
        eprintln!(
            "[assign] estimated_imbalance={:.3} group_sums=[{}] group_sizes=[{}]",
            initial_imbalance,
            format_group_sums(&state.group_sum),
            format_group_sizes(&state.groups)
        );
    }

    let refine_stats = refine_assignment(&mut judge, &mut state);
    let queries_after_refine = judge.used;
    let final_imbalance = estimated_imbalance(&state.group_sum);
    if log_enabled {
        eprintln!(
            "[refine] iterations={} balance_queries={} candidate_queries={} accepted_moves={} rejected_moves={} equal_group_comparisons={} no_move_rounds={} estimated_imbalance={:.3} queries_used={}/{}",
            refine_stats.iterations,
            refine_stats.balance_queries,
            refine_stats.candidate_queries,
            refine_stats.accepted_moves,
            refine_stats.rejected_moves,
            refine_stats.equal_group_comparisons,
            refine_stats.no_move_rounds,
            final_imbalance,
            queries_after_refine,
            judge.q
        );
        eprintln!(
            "[refine] group_sums=[{}] group_sizes=[{}]",
            format_group_sums(&state.group_sum),
            format_group_sizes(&state.groups)
        );
    }

    let final_query_stats = use_remaining_queries(&mut judge, &mut state);
    let queries_after_final_usage = judge.used;
    if log_enabled {
        eprintln!(
            "[endgame] pair_queries={} candidate_queries={} accepted_moves={} exhausted_pairs={} estimated_imbalance={:.3} queries_used={}/{}",
            final_query_stats.pair_queries,
            final_query_stats.candidate_queries,
            final_query_stats.accepted_moves,
            final_query_stats.exhausted_pairs,
            estimated_imbalance(&state.group_sum),
            queries_after_final_usage,
            judge.q
        );
        eprintln!(
            "[endgame] group_sums=[{}] group_sizes=[{}]",
            format_group_sums(&state.group_sum),
            format_group_sizes(&state.groups)
        );
    }

    let swap_stats = if ordering_mode == "insertion" {
        insertion_endgame_swaps(&mut judge, &mut state)
    } else {
        SwapStats::default()
    };
    let queries_after_swaps = judge.used;
    if log_enabled {
        eprintln!(
            "[swap] pair_queries={} swap_queries={} accepted_swaps={} stalled_rounds={} estimated_imbalance={:.3} queries_used={}/{}",
            swap_stats.pair_queries,
            swap_stats.swap_queries,
            swap_stats.accepted_swaps,
            swap_stats.stalled_rounds,
            estimated_imbalance(&state.group_sum),
            queries_after_swaps,
            judge.q
        );
        eprintln!(
            "[swap] group_sums=[{}] group_sizes=[{}]",
            format_group_sums(&state.group_sum),
            format_group_sizes(&state.groups)
        );
    }

    judge.fill_rest();
    if log_enabled {
        eprintln!(
            "[final] filler_queries={} total_queries={}/{}",
            judge.used.saturating_sub(queries_after_swaps),
            judge.used,
            judge.q
        );
    }

    for i in 0..judge.n {
        if i > 0 {
            print!(" ");
        }
        print!("{}", state.group_of[i]);
    }
    println!();
    io::stdout().flush().unwrap();
}
