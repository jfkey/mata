#include "Application.h"
#include "Graph.h"
#include "Timer.h"
#include "Utility.h"
#include <stdlib.h>
#include <string>
#include <vector>
#include <iostream>

#ifdef __cplusplus
#define XETR extern "C"
#else
#define XETR
#endif

#ifdef _WIN32
#define LIB XETR __declspec(dllexport)
#else
#define LIB XETR
#endif

using namespace std;
ui label2int(const char *str, map<string, ui> &M) {
	if (M.find(string(str)) == M.end()) M[string(str)] = M.size();
	return M[string(str)];
}

map<string, ui> vM, eM;		// 构建图的时候，需要对节点和边排序
vector<Graph*> graphs;		//
map<string, int> id2idxM;	//


inline Graph* covertstr2graph(const char* g_str) {
	if (g_str == NULL) return NULL;
	vector<pair<int, ui> > vertices;
	vector<pair<pair<int, int>, ui> > edges;
	char buf[128], buf1[128];
	string id;
	string delim = "\n", token;
	std::stringstream ss(g_str);

	while (getline(ss, token, '\n')) {
		const char* line = token.c_str();
		if (line[0] == 't') {
			sscanf(line + 2, "%s%s", buf1, buf);
			id = string(buf);
		}
		else if (line[0] == 'v') {
			int a;
			sscanf(line + 2, "%d%s", &a, buf);
			vertices.pb(mp(a, label2int(buf, vM)));
		}
		else if (line[0] == 'e') {
			int a, b;
			sscanf(line + 2, "%d%d%s", &a, &b, buf);
			edges.pb(mp(mp(a, b), label2int(buf, eM)));
			edges.pb(mp(mp(b, a), label2int(buf, eM)));
		}
		else {
			printf("!!! Unrecongnized first letter in a line when loading DB!\n");
		}
	}

	sort(vertices.begin(), vertices.end());
	for (ui i = 0; i < vertices.size(); i++) assert(vertices[i].first == i);

	sort(edges.begin(), edges.end());
	for (ui i = 0; i < edges.size(); i++) {
		assert(edges[i].first.first >= 0 && edges[i].first.first < vertices.size());
		assert(edges[i].first.second >= 0 && edges[i].first.second < vertices.size());
		if (i > 0) assert(edges[i].first != edges[i - 1].first);
		assert(edges[i].second < eM.size());
	}
	return new Graph(id, vertices, edges);
}

/*
MATA*, i.e., A*LSa based on partially matched nodes
input:
	q_str: the string of graph q with the same format with A*LSa, e.g.,  t # 186\nv 0 C\nv 1 C\nv 2 C\nv 3 C\nv 4 C\ne 0 1 1\ne 1 2 1\ne 1 3 1\ne 2 4 1\ne 1 4 1
	g_str: the string of graph g. |q| should be less than |g|
	upper_bound_map: the node map of upper bound, which is utilized to prune the search space along with A*LSa.
					 if the first and second equal 0, then do not use the prune strategy.
	matching_order: the node matching order of the MATA*. if the first and second equal 0, then do use the default order of A*LSa
	matched_nodes: the matched nodes when running MATA*, prune search branch.
	k: topk matched nodes. if k = -1, the use all nodes as the matched nodes. if k = -2, the algorithm degenerates to A*LSa
	beam_size: the beam width of the MATA*
output:
	ged search_space time_cost q_id g_id node_matching
	e.g., 0 5 2 186 126 0|2 1|1 2|0 3|3 4|4
	time_cost: microsecond.
	node_matching: u1|v1 u2|v2 ... um|vm ... -1|vn, where u_i is the node of graph q,  and v_i is the node of graph g.
*/
LIB const char* ged(const char* q_str, const char* g_str, int* upper_bound_map, int* matching_order, int**matched_nodes, int k, int beam_size) {
	std::string lower_bound = "LSa";
	int search_space = -1, ged = INF;

	if (beam_size == -1) beam_size = INF;
	Graph *q = covertstr2graph(q_str);
	Graph *g = covertstr2graph(g_str);

	// string res_str ="", map_str = "";
	ui max_bytes = 2048;
	char *res_str = new char[max_bytes], *map_str = new char[max_bytes], *tmp_str = new char[max_bytes];
	res_str[0] = '\0', map_str[0] = '\0', tmp_str[0] = '\0';
	if (q != NULL && g != NULL) {
		if (q->n > g->n) {
			Graph *t = q;  q = g;  g = t;
		}

		if (k == -1) {
			matched_nodes = new int*[q->n];
			for (int i = 0; i < q->n; i++) {
				matched_nodes[i] = new int[g->n];
				for (int j = 0; j < g->n; j++) matched_nodes[i][j] = j;
			}
		}
		if (k == -2) {
			matched_nodes = new int*[q->n];
			for (int i = 0; i < q->n; i++) {
				matched_nodes[i] = new int[g->n];
				for (int j = 0; j < g->n; j++) matched_nodes[i][j] = j;
			}
			matching_order = new int[q->n];
			upper_bound_map = new int[q->n];
			for (int i = 0; i < q->n; i++) { matching_order[i] = i; upper_bound_map[i] = i; }
			matching_order[0] = 0; matching_order[1] = 0;
			upper_bound_map[0] = 0; upper_bound_map[1] = 0;
		}

		Timer t;
		Application *app = new Application(beam_size, lower_bound.c_str());
		app->init(q, g, upper_bound_map, matching_order, matched_nodes, k);
		ged = app->AStar();
		int time_cost = int(t.elapsed()*0.001);
		search_space = app->get_search_space();
		vector<pair<ui, ui>> mapping;
		bool isSuccess = app->get_mapping(mapping, q->n);
		if (isSuccess) {
			vector<int> v2(g->n); for (int i = 0; i < g->n; i++) v2[i] = i;
			for (int i = 0; i < q->n; i++) {
				sprintf(tmp_str, " %d|%d", mapping[i].first, mapping[i].second);
				v2[mapping[i].second] = -1;
				strcat(map_str, tmp_str);
			}
			for (int i = 0; i < g->n; i++) {
				if (v2[i] != -1) {
					sprintf(tmp_str, " %d|%d", -1, v2[i]);
					strcat(map_str, tmp_str);
				}
			}
			// also check the edit distance for the one-one mapping.
			int* matching_order = new int[q->n];
			int* upper_bound_map = new int[q->n];
			for (int i = 0; i < q->n; i++) { matching_order[i] = i; upper_bound_map[i] = i; }
			matching_order[0] = 0; matching_order[1] = 0;
			upper_bound_map[0] = 0; upper_bound_map[1] = 0;
			int k_tmp = -2, beam_size_tmp = INF, ged_tmp = 10000;
			Application * app_tmp = new Application(beam_size_tmp, lower_bound.c_str());
			app_tmp->init(q, g, upper_bound_map, matching_order, matched_nodes, k_tmp);
			for (int t = 0; t < k; t++) {
				map_str[0] = '\0', tmp_str[0] = '\0';
				for (int i = 0; i < q->n; i++) {
					app_tmp->MO[i] = i;
					app_tmp->BX[i] = matched_nodes[i][t];
					sprintf(tmp_str, " %d|%d", i, matched_nodes[i][t]);
					strcat(map_str, tmp_str);
				}
				for (int i = q->n; i < g->n; i++) {
					sprintf(tmp_str, " %d|%d", -1, i);
					strcat(map_str, tmp_str);
				}
				int comp_ged = app_tmp->compute_ged_of_BX();
				if (comp_ged < ged_tmp) ged_tmp = comp_ged;
			}
			if (ged_tmp < ged) {
				sprintf(res_str, "%d %d %d %s %s %s", ged_tmp, search_space, time_cost, q->id.c_str(), g->id.c_str(), map_str);
			}
			else {
				sprintf(res_str, "%d %d %d %s %s %s", ged, search_space, time_cost, q->id.c_str(), g->id.c_str(), map_str);
			}
			delete app, q, g;
		}
		else {			// fail to find the full mapping due to the prune conflit of mata and A* beam
			int* matching_order = new int[q->n];
			int* upper_bound_map = new int[q->n];
			for (int i = 0; i < q->n; i++) { matching_order[i] = i; upper_bound_map[i] = i; }
			matching_order[0] = 0; matching_order[1] = 0;
			upper_bound_map[0] = 0; upper_bound_map[1] = 0;
			int k_tmp = -2, beam_size_tmp = INF, ged_tmp = 10000;
			Application * app_tmp = new Application(beam_size_tmp, lower_bound.c_str());
			app_tmp->init(q, g, upper_bound_map, matching_order, matched_nodes, k_tmp);
			for (int t = 0; t < k; t++) {
				map_str[0] = '\0', tmp_str[0] = '\0';
				for (int i = 0; i < q->n; i++) {
					app_tmp->MO[i] = i;
					app_tmp->BX[i] = matched_nodes[i][t];
					sprintf(tmp_str, " %d|%d", i, matched_nodes[i][t]);
					strcat(map_str, tmp_str);
				}
				for (int i = q->n; i < g->n; i++) {
					sprintf(tmp_str, " %d|%d", -1, i);
					strcat(map_str, tmp_str);
				}
				int comp_ged = app_tmp->compute_ged_of_BX();
				if (comp_ged < ged_tmp) ged_tmp = comp_ged;

			}
			sprintf(res_str, "%d %d %d %s %s %s", ged_tmp, search_space, time_cost, q->id.c_str(), g->id.c_str(), map_str);
			delete app_tmp, q, g;
		}
	}
	else {
		string ss = "";
		sprintf(res_str, "%d %d %d %s %s %s", ged, search_space, search_space, ss.c_str(), ss.c_str(), ss.c_str());
	}
	return res_str;
}

/*
Input:
	q_str: the string of graph q with the same format with A*LSa, e.g.,  t # 186\nv 0 C\nv 1 C\nv 2 C\nv 3 C\nv 4 C\ne 0 1 1\ne 1 2 1\ne 1 3 1\ne 2 4 1\ne 1 4 1
	g_str: the string of graph g. |q| should be less than |g|
	q_order_nodes: execution order of graph q nodes
	g_order_nodes: execution order of graph g nodes
Output:
	int: compute the edit distance for the given mapping
*/
LIB int mapping_ed(const char* q_str, const char* g_str, int* q_order_nodes, int* g_order_nodes) {
	std::string lower_bound = "LSa";
	int search_space = -1, ged = INF;
	Graph *q = covertstr2graph(q_str);
	Graph *g = covertstr2graph(g_str);
	if (q != NULL && g != NULL) {
		if (q->n > g->n) {
			Graph *t = q;  q = g;  g = t;
		}
	}
	//only for initilization.
	int** matched_nodes = new int*[q->n];
	for (int i = 0; i < q->n; i++) {
		matched_nodes[i] = new int[g->n];
		for (int j = 0; j < g->n; j++) matched_nodes[i][j] = j;
	}
	int* matching_order = new int[q->n];
	int* upper_bound_map = new int[q->n];
	for (int i = 0; i < q->n; i++) { matching_order[i] = i; upper_bound_map[i] = i; }
	matching_order[0] = 0; matching_order[1] = 0;
	upper_bound_map[0] = 0; upper_bound_map[1] = 0;
	int k = -2;
	//
	Timer t;
	int beam_size = INF;
	Application *app = new Application(beam_size, lower_bound.c_str());
	app->init(q, g, upper_bound_map, matching_order, matched_nodes, k);
	//////////////////////////////////////////////////////////
	for (int i = 0; i < q->n; i++ ) {
		app->MO[i] = q_order_nodes[i];
		app->BX[i] = g_order_nodes[i];
	}
	int comp_ged = app->compute_ged_of_BX();
	return comp_ged;
}

//ged; insert_node; insert_edge; remove edge, relabel;
LIB const char* map_operations(const char* q_str, const char* g_str, int* q_order_nodes, int* g_order_nodes) {
	ui max_bytes = 2048;
	char *res_str = new char[max_bytes];
	std::string lower_bound = "LSa";
	int search_space = -1, ged = INF;
	Graph *q = covertstr2graph(q_str);
	Graph *g = covertstr2graph(g_str);
	if (q != NULL && g != NULL) {
		if (q->n > g->n) {
			Graph *t = q;  q = g;  g = t;
		}
	}
	//only for initilization.
	int** matched_nodes = new int*[q->n];
	for (int i = 0; i < q->n; i++) {
		matched_nodes[i] = new int[g->n];
		for (int j = 0; j < g->n; j++) matched_nodes[i][j] = j;
	}
	int* matching_order = new int[q->n];
	int* upper_bound_map = new int[q->n];
	for (int i = 0; i < q->n; i++) { matching_order[i] = i; upper_bound_map[i] = i; }
	matching_order[0] = 0; matching_order[1] = 0;
	upper_bound_map[0] = 0; upper_bound_map[1] = 0;
	int k = -2;
	//
	Timer t;
	int beam_size = INF;
	Application *app = new Application(beam_size, lower_bound.c_str());
	app->init(q, g, upper_bound_map, matching_order, matched_nodes, k);
	//////////////////////////////////////////////////////////
	for (int i = 0; i < q->n; i++) {
		app->MO[i] = q_order_nodes[i];
		app->BX[i] = g_order_nodes[i];
	}
	int* comp_op = app->compute_operation_of_BX();
	sprintf(res_str, "%d %d %d %d %d", comp_op[0], comp_op[1], comp_op[2], comp_op[3], comp_op[4]);
	return res_str;
}


// g++ -shared -Wl,-soname,mata -o mata.so -fPIC Mata.cpp Application.cpp
