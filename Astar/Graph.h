#ifndef _GRAPH_H_
#define _GRAPH_H_

#include "Utility.h"

class Graph {
public:
	std::string id;
	ui n, m;		// n 节点的个数，m 边的个数
	ui *pstarts, *edges;		// pstarts 边的起始点。
	ui *vlabels, *elabels;

public:
	Graph(const std::string &_id, const std::vector<std::pair<int, ui> > &_vertices, const std::vector<std::pair<std::pair<int, int>, ui> > &_edges) {
		id = _id;
		n = _vertices.size();
		m = _edges.size();

		pstarts = new ui[n + 1]; vlabels = new ui[n];
		edges = new ui[m]; elabels = new ui[m];

		for (ui i = 0; i < n; i++) vlabels[i] = _vertices[i].second;

		for (ui i = 0; i < m; i++) {
			edges[i] = _edges[i].first.second;
			elabels[i] = _edges[i].second;
			//assert(elabels[i] >= 0&&elabels[i] < 3);
		}

		ui idx = 0;
		pstarts[0] = idx;
		for (ui i = 0; i < n; i++) {
			while (idx < m&&_edges[idx].first.first == i) ++idx;
			pstarts[i + 1] = idx;
		}
		assert(pstarts[n] == m);
	}

	~Graph() {
		if (pstarts != nullptr) {
			delete[] pstarts;
			pstarts = nullptr;
		}
		if (edges != nullptr) {
			delete[] edges;
			edges = nullptr;
		}
		if (vlabels != nullptr) {
			delete[] vlabels;
			vlabels = nullptr;
		}
		if (elabels != nullptr) {
			delete[] elabels;
			elabels = nullptr;
		}
	}

	void write_graph(FILE *fout, const std::vector<std::string> &_vlabels, const std::vector<std::string> &_elabels, bool bss) {
		assert(fout != NULL);
		if (bss) {
			for (ui i = 0; i < id.length(); i++) if (id[i] < '0' || id[i] > '9') printf("!!! Wrong graph id for bss\n");
			fprintf(fout, "%s\n", id.c_str());
			fprintf(fout, "%d %d\n", n, m / 2);
			for (ui i = 0; i < n; i++) fprintf(fout, "%d\n", vlabels[i]);
			for (ui i = 0; i < n; i++) for (ui j = pstarts[i]; j < pstarts[i + 1]; j++) if (edges[j] > i) {
				fprintf(fout, "%u %u %d\n", i, edges[j], elabels[j]);
			}
		}
		else {
			fprintf(fout, "t # %s\n", id.c_str());
			for (ui i = 0; i < n; i++) fprintf(fout, "v %u %s\n", i, _vlabels[vlabels[i]].c_str());
			for (ui i = 0; i < n; i++) for (ui j = pstarts[i]; j < pstarts[i + 1]; j++) if (edges[j] > i) {
				fprintf(fout, "e %u %u %s\n", i, edges[j], _elabels[elabels[j]].c_str());
			}
		}
	}

	bool is_connected() {
		std::vector<ui> Q;
		char *vis = new char[n];
		memset(vis, 0, sizeof(char)*n);
		vis[0] = 1;
		Q.push_back(0);
		for (ui i = 0; i < Q.size(); i++) {
			for (ui j = pstarts[Q[i]]; j < pstarts[Q[i] + 1]; j++)
				if (!vis[edges[j]]) {
					vis[edges[j]] = 1;
					Q.push_back(edges[j]);
				}
		}

		if (Q.size() == n) return 1;
		return 0;
	}

	ui size_based_bound(Graph *g) {
		ui r1 = n > g->n ? n - g->n : g->n - n;
		ui r2 = m > g->m ? m - g->m : g->m - m;
		assert(r2 % 1 == 0);
		return r1 + r2 / 2;
	}
	 
};

#endif
