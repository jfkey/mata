#ifndef _APPLICATION_H_
#define _APPLICATION_H_

#include <algorithm>
#include "Utility.h"
#include "Graph.h"

enum LB_Method { LSa, BMa, BMao };

struct State {
	State *parent;			// 当前状态的父亲节点状态
	State *pre_sibling;		// previous sibling
	ushort level;			// 当前的层次
	ushort image;				// 匹配的点？ 
	ushort mapped_cost;			// 匹配的代价
	ushort lower_bound;			// 估计的代价
	ushort cs_cnt; // cs_cnt stores how many children and right sibling depends on this state; this state can be removed if and only cs_cnt becomes 0
#ifdef _EXPAND_ALL_
	ushort *siblings; // remaining siblings			// 当前层剩余的兄弟节点
	ushort siblings_n;
#endif
};

class Application {
	//private:
public:
	ui q_n;			// q 表示第一个图 （较小的图）
	ui *q_starts, *q_edges;
	ui *q_vlabels, *q_elabels;

	ui g_n;			// g 表示第二个图 （较大的图）
	ui *g_starts, *g_edges;
	ui *g_vlabels, *g_elabels;   // g_vlabels: 第二个图中，每个节点的标签

	ui vlabels_n, elabels_n;

	ui beam_size;
	ui verify_upper_bound;
	ui upper_bound;

	LB_Method lb_method;

	bool q_g_swapped;		// 判断 第一、二图是否交换

	
	ui search_n;
	ui search_n_for_IS;

	char *visX, *visY;		// visX: 标记q中的哪些节点已经被匹配
	int *mx, *my;
	ui *BX;
	ui *candidates;
	ui *queue, *prev;

	std::vector<State *> states_memory;				// 存放状态池
	std::vector<State *> states_pool;				// 状态池（初始化1024个状态），
	ui states_pool_n;								// 状态池中剩余的状态个数

	std::vector<ushort *> siblings_memory;			// 存放 兄弟节点池
	std::vector<ushort *> siblings_pool;			// 兄弟节点池
	ui siblings_pool_n;								// 兄弟节点池中剩余节点的个数

	int *elabels_map, *vlabels_map;					// 论文中提到的两个数据结构，注意是g中未匹配的子图 - q中未匹配的子图（节点u不在此子图中）
	short *elabels_matrix;
	uchar *q_matrix;		// TODO:

	ushort *visited_siblings; // these two arrays only used in DFS
	ushort *visited_siblings_n;

	/***** for bipartite matching based lower bounds ****/
	ui *cost;
	int *lx, *ly, *slack, *slackmy;

	std::pair<int, int> *children;

	long long search_space;
	
	int *MO;									// 匹配的节点顺序，matching order. ui * MO. 
	int** matched_nodes;						// topK matched nodes for each node
	int k;										// topK
	bool **in_matched_nodes;					// 根据matched_nodes构建，表示每个节点，i,j 是否匹配

public:
	Application(ui _beam_size, const char *lower_bound);
	~Application();

	long long get_search_space() { return search_space; }

	void init(const std::vector<std::pair<int, int> > &g_v, const std::vector<std::pair<std::pair<int, int>, int> > &g_e, const std::vector<std::pair<int, int> > &q_v, const std::vector<std::pair<std::pair<int, int>, int> > &q_e);
	void init(const Graph *g, const Graph *q, int* ub_map, int* MO, int**matched_nodes, int k);
	ui DFS(State *node = NULL);
	ui AStar(std::vector<std::pair<ui, ui> > *mapping_ptr = NULL, int *lb_ptr = NULL);
	ui compute_ged_of_BX();
	int* compute_operation_of_BX();
	bool get_mapping(std::vector<std::pair<ui, ui> > &mapping, ui n);

public:
	void preprocess();

	void add_to_pool(State *st);
	void add_to_heap(State *st, ui &heap_n, std::vector<State*> &heap);
	State* get_a_new_state_node();
	void put_a_state_to_pool(State *st);
	ushort* get_a_new_siblings_node();
	void put_a_sibling_to_pool(ushort *sibling);
	void verify_induced_ged(State *now);
	void verify_LS_lower_bound(State *now);
	ui relabel(ui len1, ui *array1, ui len2, ui *array2);
	ui search_index(ui val, std::vector<ui> &array, ui array_len);

	void compute_mapping_order();
	void generate_best_extension(State *parent, State *now); // compute the best child of a state
	void compute_mapped_cost_and_upper_bound(State *now, ui n, ui *candidates, int *mapping);
	void construct_sibling(State *pre_sibling, State *now); // compute the best ungenerated sibling of a state
	void extend_to_full_mapping(State *parent, State *now);
	void compute_mapped_cost(State *now);


	void compute_best_extension_LSa(State *now, ui candidate_n, ui *candidates, ui pre_siblings);
	void compute_best_extension_BM(char anchor_aware, State *now, ui candidate_n, ui *candidates, ui pre_siblings, char no_siblings, char IS = 0);
	void compute_best_extension_BMa(char anchor_aware, State *now, ui n, ui *candidates, ui pre_siblings);

	ui Hungarian(char initialization, ui n, ui *cost); // minimum cost bipartite matching
	void heap_top_down(ui idx, ui heap_n, std::vector<std::pair<double, int> > &heap, ui *pos);
	void heap_bottom_up(ui idx, std::vector<std::pair<double, int> > &heap, ui *pos);
	void heap_top_down(ui idx, ui heap_n, std::vector<State*> &heap);
	void heap_bottom_up(ui idx, std::vector<State*> &heap);

	ui compute_ub_map(int *ub_map);
	void print_heap(ui heap_n, std::vector<State*> &heap);
	int find_first_match(std::pair<int, int> *children, int size, bool *cur_matched_nodes);
};

#endif
