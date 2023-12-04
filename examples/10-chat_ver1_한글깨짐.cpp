//#pragma warning (disable : 4996)
//#pragma warning (disable : 6031)
#include <iostream>
#include <stdio.h>
#include <cstdio>
#include <stdlib.h>
#include <cstdbool>
#include <string>
#include <cstring>
#include <time.h>
#include "openai.hpp"
#include "nlohmann/json.hpp"
#define _CRT_SECURE_NO_WARNINGS

using namespace std;

const int INF = 987654321;     // INF�� 9��.


/*
[05.17]
1. �׷���(100,300)�� ���� + ���� �ߺ����� 300�� �������� + ����׷��� Ȯ�α��� �Ϸ�
	- ������� 100x100�̹Ƿ� memory : 40kb

[05.18]
1. ������ ���� ����ġ(1~9)�� ����. ���ð� �Ÿ� distance�� �ǹ�.
2. �÷��̵� ���� �˰������� ���ð� �̵��� ��, �׻� �ִܰŸ��� �̵��ϵ��� ��������� ������Ʈ ����.
3. RBƮ���� �⺻ �Լ����� report2���� �״�� ��������.
4. ȣ��RBƮ���� ������. (key : ȣ�� ����)
*/
#define NODES (100)     // ���� ���� : 100��
#define EDGES (300)     // transportation ����(�װ��� ����) : 300��
#define HOTELS (100)     // �� ���ô� ȣ�� ���� : 100���� (�� 10,000��)




//�ѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤ� RBƮ�� �Լ��� ���� �ѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤ�//

enum Color {
	RED = 0,
	BLACK = 1
};

// ��� ����ü
struct node_t {
	// key, left, right, parent, color
	int key;
	struct node_t* left;
	struct node_t* right;
	struct node_t* parent;
	enum Color color;
};

struct rbtree {
	struct node_t* root;
	struct node_t* NIL;
};

// leaf = nil ���
node_t create_nil = {
	/* .key = */ -1,
	/* .left = */ nullptr,
	/* .right = */ nullptr,
	/* .parent = */ nullptr,
	/* .color = */ BLACK
};

node_t* nil = &create_nil;


struct trunk {
	struct trunk* prev;
	string str;
};

// rand()�� �ߺ��Ǹ� true ����
bool is_duplicate(int* list, int value, int end) {

	for (int i = 0; i < end; i++) {
		if (list[i] == value) {
			return true;
		}
	}
	return false;
}

// ��� ����
void del_node(struct node_t* n) {
	if (n != NULL) {
		del_node(n->left);
		del_node(n->right);
	}
	free(n);
}

// rbƮ�� ��ü �ʱ�ȭ
void delete_rbtree(struct rbtree* tree) {
	del_node(tree->root);
	free(tree);
}

// ���ο� ��� ����
struct node_t* newNode(struct rbtree* tree, int k) {
	struct node_t* new_node = (struct node_t*)malloc(sizeof(struct node_t));

	new_node->parent = NULL;
	new_node->left = tree->NIL;
	new_node->right = tree->NIL;
	new_node->key = k;
	new_node->color = RED;

	return new_node;
}

// rbƮ�� ���� (��Ʈ �޸� �Ҵ�)
struct rbtree* newRBtree() {
	struct rbtree* tree = (struct rbtree*)malloc(sizeof(struct rbtree));
	struct node_t* temp_nil = (struct node_t*)malloc(sizeof(struct node_t));

	temp_nil->left = NULL;
	temp_nil->right = NULL;
	temp_nil->parent = NULL;
	temp_nil->color = BLACK;
	temp_nil->key = -1;

	tree->NIL = temp_nil;
	tree->root = temp_nil;

	return tree;
}

struct node_t* node_nil(struct node_t* x) {
	if (x == NULL) {
		return nil;
	}
	else {
		return x;
	}
}

// RBƮ������ �ּ� key ã�� : cheapest hotel reserve
struct node_t* MinKeyOfRBtree(struct rbtree* t) {
	if (t->root == t->NIL) {
		return NULL;
	}
	struct node_t* curr = t->root;

	while (curr->left != t->NIL) {
		curr = curr->left;
	}
	return curr;
}

// RBƮ������ �ִ� key ã�� : flex hotel reserve
struct node_t* MaxKeyOfRBtree(struct rbtree* t) {
	if (t->root == t->NIL) {
		return NULL;
	}
	struct node_t* curr = t->root;

	while (curr->right != t->NIL) {
		curr = curr->right;
	}
	return curr;
}

// SUCCESSOR ã��
struct node_t* tree_successor(struct rbtree* t, struct node_t* n) {
	// right child�� ������ ���, ���� ����� ���� ������ ã���� ��
	// right sub tree�� �ּڰ�
	if (n->right != NULL) {
		// temp�� NULL�� �ɶ����� �������� �̵�
		struct node_t* temp_c = n->right;
		while (temp_c->left != NULL) {
			temp_c = temp_c->left;
		}
		return temp_c;
	}
	// right node�� ������, n�� parent ���� ��
	struct node_t* temp_p = n->parent;
	// ���� �ö󰡸鼭 temp node�� left child�� �ɶ����� �ݺ�
	// �������� �� �ö󰡴ٰ� ���������� ó������ �ö��� �� �� ���� successor�̴�.
	while (temp_p != NULL && n == temp_p->right) {
		n = temp_p;
		temp_p = temp_p->parent;
	}

	return temp_p;
}

// PREDECCESSOR ã��
struct node_t* tree_predeccessor(struct rbtree* t, struct node_t* n) {
	// left child�� ������ ���, ���� ����� ���� �������� ã���� ��
	// Left sub tree�� �ִ�
	if (n->left != t->NIL) {
		// temp�� NULL�� �ɶ����� �������� �̵�
		struct node_t* temp_c = n->left;
		while (temp_c->right != t->NIL) {
			temp_c = temp_c->right;
		}
		return temp_c;
	}
	// left node�� ������, n�� parent ���� ��
	struct node_t* temp_p = n->parent;
	// ���� �ö󰡸鼭 temp node�� right child�� �ɶ����� �ݺ�
	// ������ �� Ÿ��ö󰡴ٰ� ó������ �������� �ö��� ��, �׳��� PREDECCESSOR�̴�.
	while (temp_p != NULL && n == temp_p->left) {
		n = temp_p;
		temp_p = temp_p->parent;
	}

	return temp_p;
}

// ���� ȸ��
void right_rotation(struct rbtree* tree, struct node_t* x) {
	// TODO!

	struct node_t* y;

	// 1. target�� left���� y ����
	y = x->left;
	// 2. y�� ������ ����Ʈ���� target�� ���� ����Ʈ���� �ű�
	x->left = y->right;
	// 3. y�� ������ ��尡 NIL�� �ƴ϶��, y�� ������ ��� �θ� target���� ����
	if (y->right != tree->NIL) {
		y->right->parent = x;
	}
	// 4. y�� �θ� ��带 target�� �θ� ���� ����
	y->parent = x->parent;
	// 5. target�� �θ� ��尡 nil�̶��, Ʈ�� ����ü�� root�� y�� ����
	if (x->parent == tree->NIL)
		tree->root = y;
	// 6. target�� target �θ� ����� �����̸�, target �θ��� ������ y�� ����
	else if (x == x->parent->left)
		x->parent->left = y;
	// 7. target�� target �θ� ����� �������̸�, target �θ��� �������� y�� ����
	else
		x->parent->right = y;
	// 8. target�� y�� ���������� ����
	y->right = x;
	// 9. target�� �θ� y�� ����
	x->parent = y;
}

void left_rotation(struct rbtree* tree, struct node_t* x) {
	// TODO!
	struct node_t* y;

	y = x->right;
	x->right = y->left;

	if (y->left != tree->NIL)
		y->left->parent = x;

	y->parent = x->parent;

	if (x->parent == tree->NIL)
		tree->root = y;
	else if (x == x->parent->left)
		x->parent->left = y;
	else
		x->parent->right = y;

	y->left = x;
	x->parent = y;
}



struct node_t* rbtree_find(const struct rbtree* t, const int key) {
	// TODO: implement find
	struct node_t* current = t->root;

	while (current != t->NIL) {
		if (current->key == key)
			return current;

		if (current->key < key)
			current = current->right;
		else
			current = current->left;
	}

	return NULL;
}

// search' �� find ���� (����� ����)
struct node_t* RB_find_ver2(struct rbtree* tree, int key) {
	struct node_t* temp;
	temp = tree->root;

	while (temp != NULL) {
		if (key == temp->key) {
			return temp;
		}
		else if (key > temp->key) {
			temp = temp->right;
		}

		else if (key < temp->key) {
			temp = temp->left;
		}
	}

	return NULL;
}

void RB_insert_fixup(struct rbtree* t, struct node_t* z) {
	struct node_t* y;

	while (z->parent->color == RED) {
		// z�� �θ� ���θ��� ���� ���� Ʈ���� ���
		if (z->parent == z->parent->parent->left) {
			y = z->parent->parent->right;

			// CASE 1 : ��� z�� ���� y�� ������ ���
			if (y->color == RED) {
				z->parent->color = BLACK;
				y->color = BLACK;
				z->parent->parent->color = RED;
				z = z->parent->parent;
			}
			// CASE 2 : z�� ���� y�� ����̸��� z�� ������ �ڽ��� ���
			else {
				if (z == z->parent->right) {
					z = z->parent;
					left_rotation(t, z);
				}
				// CASE 3 : z�� ���� y�� ����̸��� z�� ������ �ڽ��� ���
				z->parent->color = BLACK;
				z->parent->parent->color = RED;
				right_rotation(t, z->parent->parent);
			}
		}
		// z�� �θ� ���θ��� ���� ���� Ʈ���� ���
		else {
			y = z->parent->parent->left;

			// CASE 4 : ��� z�� ���� y�� ������ ���
			if (y->color == RED) {
				z->parent->color = BLACK;
				y->color = BLACK;
				z->parent->parent->color = RED;
				z = z->parent->parent;
			}
			// CASE 5 : z�� ���� y�� ����̸��� z�� ������ �ڽ��� ���
			else {
				if (z == z->parent->left) {
					z = z->parent;
					right_rotation(t, z);
				}
				// CASE 6 : z�� ���� y�� ����̸��� z�� ������ �ڽ��� ���
				z->parent->color = BLACK;
				z->parent->parent->color = RED;
				left_rotation(t, z->parent->parent);
			}
		}
	}
	t->root->color = BLACK;
}

// ��� ����
void RB_INSERT(struct rbtree* T, int k) {
	// NIL�� root node
	struct node_t* y = T->NIL;
	struct node_t* x = T->root;

	// ���ο� ��� ����
	struct node_t* z = newNode(T, k);



	// BST Ʈ���� ���� ������� ��� ����
	while (x != T->NIL) {
		y = x;
		if (k < x->key) {
			x = x->left;
		}
		else {
			x = x->right;
		}
	}

	z->parent = y;

	if (y == T->NIL) {
		T->root = z;
	}
	else if (k < y->key) {
		y->left = z;
	}
	else {
		y->right = z;
	}

	// Fixup ����
	RB_insert_fixup(T, z);
}


void rbtree_transplant(struct rbtree* t, struct node_t* u, struct node_t* v) {
	if (u->parent == t->NIL) {
		t->root = v;
	}
	else if (u == u->parent->left) {
		u->parent->left = v;
	}
	else {
		u->parent->right = v;
	}

	v->parent = u->parent;
}

void rbtree_delete_fixup(struct rbtree* t, struct node_t* x) {
	while (x != t->root && x->color == BLACK) {
		// CASE 1 ~ 4 : LEFT CASE
		if (x == x->parent->left) {
			struct node_t* w = x->parent->right;

			// CASE 1 : x�� ���� w�� ������ ���
			if (w->color == RED) {
				w->color = BLACK;
				x->parent->color = RED;
				left_rotation(t, x->parent);
				w = x->parent->right;
			}

			// CASE 2 : x�� ���� w�� ����̰� w�� �� ������ ��� ����� ���
			if (w->left->color == BLACK && w->right->color == BLACK) {
				w->color = RED;
				x = x->parent;
			}

			// CASE 3 : x�� ���� w�� ���, w�� ���� �ڽ��� ����, w�� ������ �ڽ��� ����� ���
			else {
				if (w->right->color == BLACK) {
					w->left->color = BLACK;
					w->color = RED;
					right_rotation(t, w);
					w = x->parent->right;
				}

				// CASE 4 : x�� ���� w�� ����̰� w�� ������ �ڽ��� ������ ���
				w->color = x->parent->color;
				x->parent->color = BLACK;
				w->right->color = BLACK;
				left_rotation(t, x->parent);
				x = t->root;
			}
		}
		// CASE 5 ~ 8 : RIGHT CASE
		else {
			struct node_t* w = x->parent->left;

			// CASE 5 : x�� ���� w�� ������ ���
			if (w->color == RED) {
				w->color = BLACK;
				x->parent->color = RED;
				right_rotation(t, x->parent);
				w = x->parent->left;
			}

			// CASE 6 : x�� ���� w�� ����̰� w�� �� ������ ��� ����� ���
			if (w->right->color == BLACK && w->left->color == BLACK) {
				w->color = RED;
				x = x->parent;
			}

			// CASE 7 : x�� ���� w�� ���, w�� ���� �ڽ��� ����, w�� ������ �ڽ��� ����� ���
			else
			{
				if (w->left->color == BLACK) {
					w->right->color = BLACK;
					w->color = RED;
					left_rotation(t, w);
					w = x->parent->left;
				}

				// CASE 8 : x�� ���� w�� ����̰� w�� ������ �ڽ��� ������ ���
				w->color = x->parent->color;
				x->parent->color = BLACK;
				w->left->color = BLACK;
				right_rotation(t, x->parent);
				x = t->root;
			}
		}
	}

	x->color = BLACK;
}

int RB_DELETE(struct rbtree* t, int k) {
	// TODO: implement erase
	struct node_t* y;
	struct node_t* x;
	int yOriginalColor;

	struct node_t* p = rbtree_find(t, k);

	y = p;
	yOriginalColor = y->color;

	if (p->left == t->NIL) {
		x = p->right;
		rbtree_transplant(t, p, p->right);
	}
	else if (p->right == t->NIL) {
		x = p->left;
		rbtree_transplant(t, p, p->left);
	}
	else {
		y = p->right;
		while (y->left != t->NIL) {
			y = y->left;
		}
		yOriginalColor = y->color;
		x = y->right;

		if (y->parent == p) {
			x->parent = y;
		}
		else {
			rbtree_transplant(t, y, y->right);
			y->right = p->right;
			y->right->parent = y;
		}

		rbtree_transplant(t, p, y);
		y->left = p->left;
		y->left->parent = y;
		y->color = p->color;
	}

	if (yOriginalColor == BLACK) {
		rbtree_delete_fixup(t, x);
	}

	free(p);

	return 0;
}



// 3-4. SEARCH�Լ� �����ϱ�
// 1. key���� �����ϸ� key ����
// 2. key���� ������ key�� successor ����
// 3. successor�� ������ Nil����
struct node_t* SEARCH(struct rbtree* tree, int k) {
	struct node_t* result = RB_find_ver2(tree, k);		// �ϴ� k �ִ��� ã�ƺ���

	if (result->key == k) {		// 1. key�� ����
		return result;
	}
	else if (result->key > k) {		// 2. key���� ������ ���� �θ� �ٷ� successor�ΰ��
		return result;
	}
	else if (result->key < k) {		// 2. key�� ����, �θ� key���� ������, �θ��� successor�� ����
		struct node_t* succ_result = tree_successor(tree, result);
		if (succ_result == NULL)			return NULL;
		else if (succ_result->key > k)		return succ_result;
		else								return NULL;			// 3. successor�� k���� �۴ٸ� = ����x��� �ǹ� = nil�� ����
	}

	return NULL;
}


void show_trunks(struct trunk* p) {
	if (p == NULL) {
		return;
	}
	show_trunks(p->prev);
	cout << p->str;
}
void PRINT_BST(struct node_t* n, struct trunk* prev, bool is_left) {
	if (n == NULL) {
		return;
	}
	string prev_str = "    ";

	//struct trunk* trunk_temp = (struct trunk*)malloc(sizeof(struct trunk));
	struct trunk* trunk_temp = new struct trunk;
	trunk_temp->prev = prev;
	trunk_temp->str = prev_str;

	PRINT_BST(n->right, trunk_temp, true);
	if (n->key >= 0) {	// nil��� ������� �ʱ� ���� : nil.key = -1�� �ʱ�ȭ �Ǿ�����
		if (!prev) {
			trunk_temp->str = "---";
		}
		else if (is_left) {
			trunk_temp->str = ",---";
			prev_str = "   |";
		}
		else {
			trunk_temp->str = "`---";
			prev->str = prev_str;
		}
		show_trunks(trunk_temp);

		printf(" %d", n->key);
		if (n->color == RED) {
			printf("%s\n", "��");
		}
		else {
			printf("%s\n", "��");
		}

		if (prev) {
			prev->str = prev_str;
		}
		trunk_temp->str = "   |";
	}
	PRINT_BST(n->left, trunk_temp, false);
}


//�ѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤ� RBƮ�� �Լ��� �� �ѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤ�//




// �÷��̵� ������ ���ؼ� ��������� ���Ѵ�� �ʱ�ȭ
void adj_init_INF(int** adjMatrix)
{
	for (int i = 0; i < NODES; i++)
		for (int j = 0; j < NODES; j++) {
			adjMatrix[i][j] = INF;
			if (i == j) adjMatrix[i][j] = 0;
		}
}

// ��� ���� -> ��� ������ �ִܰ�θ� �˱� ���ؼ� �÷��̵���� �˰��� ���
// ȿ�� : �� ���ø� �̵��� ������ �׻� �ִܰ�η� �̵��ϰ� ��.
void floyd(int** adjMatrix)
{
	for (int m = 0; m < NODES; m++) {       // ��������
		for (int s = 0; s < NODES; s++) {       // ���۵���
			for (int e = 0; e < NODES; e++) {       // ��������
				// �߰���ΰ� �Ұ����ϸ� �ǳʶ�!
				if (adjMatrix[s][m] == INF || adjMatrix[m][e] == INF)
					continue;
				//����� ���İ��� ���� �� ������ �װɷ� ������Ʈ�Ѵ�.
				if (adjMatrix[s][e] > adjMatrix[s][m] + adjMatrix[m][e])
					adjMatrix[s][e] = adjMatrix[s][m] + adjMatrix[m][e];
			}
		}
	}
}

// ���� ����� �����ϰ� �ʱ�ȭ�ϴ� �Լ�
int** createAdjMatrix() {
	int** adjMatrix = (int**)malloc((NODES + 5) * sizeof(int*));
	for (int i = 0; i < NODES; i++) {
		adjMatrix[i] = (int*)malloc((NODES + 5) * sizeof(int));       // �����Ӱ� adj[105][105]�� ������� ����
		for (int j = 0; j < NODES; j++) {
			adjMatrix[i][j] = 0;  // �ʱⰪ�� �ϴ� 0���� ���� -> ���߿� INF�� �ʱ�ȭ(�÷��̵�)
		}
	}
	return adjMatrix;
}

// ���� ��� �޸� ����
void freeAdjMatrix(int** adjMatrix) {
	for (int i = 0; i < NODES + 5; i++) {
		free(adjMatrix[i]);
	}
	free(adjMatrix);
}

// �� ��� ���̿� ������ �߰��ϴ� �Լ� : ������ ����ġ�� ���� ������ �Ÿ��� �ǹ���.
void addEdge(int** adjMatrix, int city1, int city2, int weight) {
	adjMatrix[city1][city2] = weight;
	adjMatrix[city2][city1] = weight;       // ����1 ~ ����2 ������ �Ÿ� �Է�
}

// DFS (���� �켱 Ž��) �Լ� : �����Լ����� Ȯ��
void dfs(int** adjMatrix, bool visited[], int city) {
	visited[city] = true;

	for (int i = 0; i < NODES; i++) {
		if (adjMatrix[city][i] < INF && !visited[i]) {  // INF���� �������̸� ���� ����� ����.
			dfs(adjMatrix, visited, i);
		}
	}
}

// �׷����� ���� �׷������� Ȯ���ϴ� �Լ�
bool isConnectedGraph(int** adjMatrix) {
	bool visited[NODES];
	for (int i = 0; i < NODES; i++) {
		visited[i] = false;
	}

	// DFS (���� �켱 Ž��)
	dfs(adjMatrix, visited, 0); // 0�����ú��� �湮 ����

	// ��� ��带 �湮�ߴ��� Ȯ��
	for (int i = 0; i < NODES; i++) {
		if (!visited[i]) {
			return false;
		}
	}

	return true;
}


// �������� �ּҰŸ�(�÷��̵� ��)������ ����ϴ� �Լ�
void printAdjMatrix(int** adjMatrix) {
	for (int i = 0; i < NODES; i++) {
		for (int j = 0; j < NODES; j++) {
			(adjMatrix[i][j] == INF) ? printf("- ") : printf("%d ", adjMatrix[i][j]);
		}
		printf("\n");
	}
}

// ���� ���� ����ϴ� �Լ�
void printOriginMatrix(int originMatrix[][NODES]) {
	for (int i = 0; i < NODES; i++) {
		for (int j = 0; j < NODES; j++) {
			(originMatrix[i][j] == INF) ? printf("- ") : printf("%d ", originMatrix[i][j]);
		}
		printf("\n");
	}
}

// ���� �׷��� G(100, 300) ����
void Make_Coneected_Graph(int** adjMatrix) {
	while (true) {
		// ���� ��� INF�� �ʱ�ȭ
		adj_init_INF(adjMatrix);

		int edgeCount = 0;
		while (edgeCount < EDGES) {
			int city1 = rand() % NODES;
			int city2 = rand() % NODES;     // �װ��� ������ ���ø� ���Ƿ� ����(300���� �װ���)
			int distance = (rand() % 900) + 100;        // ���ð� �Ÿ� : 100~999�� ������

			// �ߺ�(�̹� ����?)�� �������� Ȯ�� �� ��ü ���� Ȯ��
			if (city1 != city2 && adjMatrix[city1][city2] == INF && adjMatrix[city2][city1] == INF) {
				addEdge(adjMatrix, city1, city2, distance);
				edgeCount++;
			}
		}

		if (isConnectedGraph(adjMatrix)) {
			break;  // ���� �׷����� �����Ǿ����� �ݺ� ����
		}
	}
}

// ȣ��RBƮ�� �����ϱ�
void Make_Hotel_RBtrees(struct rbtree* HT[NODES]) {
	for (int i = 0; i < NODES; i++) {
		HT[i] = newRBtree();  // ���ø��� ȣ�� ��Ʈ�� �����Ͽ� �迭�� �Ҵ�
	}

	// ȣ�ڷ�Ʈ���� 100���� ȣ�� ����(�� 10,000�� ����) : ������ ������ key
	// ������ ���� ���� ���� : 10���� . 10��100�� .....49��9900�� (10��~49��9900��. 100��������) + �ߺ��� ����

	int random_price;
	// identical value ������ ���� üũ�� ����Ʈ
	int* A = (int*)malloc(HOTELS * sizeof(int));

	// �� ���ø��� ȣ��100���� ����
	for (int i = 0; i < NODES; i++) {	// i ���ø���
		for (int j = 0; j < HOTELS; j++)	A[j] = 0;	// �ߺ� üũ �迭 �ʱ�ȭ

		// 100���� ���� ���� insert
		for (int j = 0; j < HOTELS; j++) {		// j ȣ��
			do {
				random_price = (rand() % 40 + 10) * 10000 + (rand() % 10) * 1000 + (rand() % 10) * 100;  // 10����~49��9900��(1000�� ����)
			} while (is_duplicate(A, random_price, j));
			A[j] = random_price;
			// A[i]�� ���� ����
			// key�� RB tree�� ����
			RB_INSERT(HT[i], random_price);
		}

		// ȣ�� RB tree ���
		/*
		printf("\n\n  Random KEY (ȣ��)�� 100��  :  ");
		for (int i = 0; i < HOTELS; i++)	printf(" %d ", A[i]);
		printf("\n\n   RBƮ�� ���!! \n\n");
		PRINT_BST(HT[i]->root, NULL, false);
		*/


	}
}

// ��� ��ҵ� �������� ����(0��° �������� ����)
void shuffle_array(int* arr, int size) {

	for (int i = size - 1; i > 0; i--) {
		// ���� �ε��� ����
		int j = (rand() % i) + 1;	// 0��°�� ������ �� ��������.

		// ���� ��ҿ� ������Ҹ� ����
		int temp = arr[i];
		arr[i] = arr[j];
		arr[j] = temp;
	}
}

// �־��� �湮������ �湮�غ��� �Ÿ��� �� ����
int Sum_distance_dest_order(int* input_dest, int num_dest, int** adjMatrix) {
	int sum = 0;
	for (int i = 0; i < num_dest - 1; i++) {
		sum += adjMatrix[input_dest[i]][input_dest[i + 1]];
	}
	// ����->���������� ���ƿ��� �Ÿ� ���������� �����ֱ�
	sum += adjMatrix[input_dest[num_dest - 1]][input_dest[0]];

	return sum;
}

// ������ ���(�湮����)¥��
int* make_dest_order(int num_dest, int* input_dest, int start, int** adjMatrix) {
	// ERROR CODE
	if (num_dest <= 0 || num_dest > 100) {
		printf("\n\n  ERROR : �������� ������ �߸��Ǿ����ϴ�.\n");
		exit(0);
	}

	// ��� ��� ����
	int* sorted_dest = (int*)malloc((num_dest + 5) * sizeof(int));

	if (num_dest <= 1) {	// 1���϶�(����ó��)
		sorted_dest[0] = start;
	}
	// �������� 1000�� �ĺ��� �ּҷ� ����
	else {	// 21~100�� : ����
		int min_cost = INF;
		int random_list[100 + 5];
		// input�� ����
		for (int i = 0; i < num_dest; i++)
			random_list[i] = input_dest[i];

		// 1000���� ���� ��θ� ���� �ּҺ�
		for (int i = 0; i < 1000; i++) {
			shuffle_array(random_list + 1, num_dest - 1);	// ����(�������� �ȼ��̵��� ����!)
			int result = Sum_distance_dest_order(random_list, num_dest, adjMatrix);		// ����غ���
			// �ּҸ� �湮���� ����
			if (result < min_cost) {
				min_cost = result;
				for (int j = 0; j < num_dest; j++)
					sorted_dest[j] = random_list[j];
			}
		}
	}

	return sorted_dest;
}

// ��ũ�� ����Ʈ�� ��� ����ü ����(��� = ����)
typedef struct linkNode {
	int cur_city;		// ���ù�ȣ
	int stay;			// �ӹ��� �� ��
	int hotel[100 + 5];	// ����� ȣ�ڰ���
	int next_city;		// ���� ���� ��ȣ
	int flight_cost;	// �װ��� ����
	struct linkNode* next;
} linkNode;

// ������ Ȯ���Ҷ� �ʿ��� ����ü
typedef struct {
	char my_name[100];			// ������ �̸�
	int year;				// ������� ��¥
	int month;
	int day;
	int period;				// ���� �Ⱓ
	int budget;				// ���
	int total_price;			// �� ����
	int total_distance;		// �� �̵��Ÿ�
	int fly_ver;			// �װ��� �ɼ�
	int hotel_ver;			// ȣ�� �ɼ�
	linkNode* head;			// �����̵� ��ũ�� ������
} Reservation_info;
Reservation_info reservation_info[1000];

// ���ο� ��带 �����ϴ� �Լ�
linkNode* createNode(int cur_city, int stay, int next_city, int flight_cost) {
	linkNode* nNode = (linkNode*)malloc(sizeof(linkNode));
	nNode->cur_city = cur_city;
	nNode->stay = stay;
	nNode->next_city = next_city;
	nNode->flight_cost = flight_cost;
	nNode->next = NULL;
	return nNode;
}

// ��ũ�� ����Ʈ�� ��带 �߰��ϴ� �Լ�
linkNode* addNode(linkNode** head, int* sorted_dest, int stay, int idx, int num_dest, int fly_ver, int** adjMatrix) {
	// ��������
	int cur_city = sorted_dest[idx];
	int next_city;
	if (idx == num_dest - 1)	next_city = sorted_dest[0];		// ������ ���ø� ó�����÷� ȸ��!
	else					next_city = sorted_dest[idx + 1];

	// �װ��� ���� ����
	int flight_cost;
	int distance = adjMatrix[cur_city][next_city];
	if (fly_ver == 1)			flight_cost = distance * 1500;		// ���ڳ��(25%����)
	else if (fly_ver == 2)		flight_cost = distance * 2000;		// ����Ͻ�(����)
	else						flight_cost = distance * 2600;		// �۽�Ʈ(30%����)


	linkNode* nNode = createNode(cur_city, stay, next_city, flight_cost);

	if (*head == NULL) {
		// ����Ʈ�� ������� ���
		*head = nNode;
	}
	else {
		// ����Ʈ�� ������� ���� ���
		linkNode* temp = *head;
		while (temp->next != NULL) {
			temp = temp->next;
		}
		temp->next = nNode;
	}

	return nNode;
}

// ��ũ�� ����Ʈ�� ����ϴ� �Լ�
void printLinkedList(linkNode* head) {
	linkNode* temp = head;
	while (temp != NULL) {
		printf("        - [ %d ]�� ���ÿ���   >   [ %d ]�� ����\n", temp->cur_city, temp->stay);
		temp = temp->next;
	}
	printf("\n");
}

// ������ ���� Ȯ�� �Լ�
void Show_reservation(int id) {
	// reservation_info[id]
			/*
			// ������ Ȯ���Ҷ� �ʿ��� ����ü
			typedef struct Reservation_info {
				char my_name[100];			// ������ �̸�
				int year;				// ������� ��¥
				int month;
				int day;
				int period;				// ���� �Ⱓ
				int budget;				// ���
				int total_price;			// �� ����
				int total_distance;		// �� �̵��Ÿ�
				int fly_ver;			// �װ��� �ɼ�
				int hotel_ver;			// ȣ�� �ɼ�
				linkNode* head;			// �����̵� ��ũ�� ������
			}reservation_info[1000];
			*/
	printf("\n\n\n  # ������ ���� #\n");
	printf("    �� ���� : %s\n", reservation_info[id].my_name);
	printf("    �� ��� ��¥ : %d�� %d�� %d��\n", reservation_info[id].year, reservation_info[id].month, reservation_info[id].day);
	printf("    �� ���� �Ⱓ : %d�ϰ�\n", reservation_info[id].period);
	printf("    �� ���� : %d (KRW)\n", reservation_info[id].budget);
	printf("    �� ��� : %d (KRW)\n", reservation_info[id].total_price);
	if (reservation_info[id].fly_ver == 1)			printf("    �� �װ� �ɼ�  :  �� [Economy] ���\n");
	else if (reservation_info[id].fly_ver == 2)		printf("    �� �װ� �ɼ�  :  �ڡ� [Business] ���\n");
	else											printf("    �� �װ� �ɼ�  :  �ڡڡ� [First] ���\n");
	if (reservation_info[id].hotel_ver == 1)				printf("    �� ȣ�� �ɼ�  :  �� [Cheapest] ȣ��\n");
	else if (reservation_info[id].hotel_ver == 2)			printf("    �� ȣ�� �ɼ�  :  �ڡ� [Resonable] ȣ��\n");
	else													printf("    �� ȣ�� �ɼ�  :  �ڡڡ� [Flex] ȣ��\n");
	printf("\n\n\n �ѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤ�[AI���� �����]�ѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤ�\n");
	printf("\n\n  (1) �ּҺ�� �װ� �̵� ��� : ");
	linkNode* head = reservation_info[id].head;
	linkNode* temp = head;
	for (temp = head; temp != NULL; temp = temp->next)
		printf(" [%d]���� -> ", temp->cur_city);
	printf(" [%d]����\n\n", head->cur_city);
	printf("        - �� �̵��Ÿ� : [ %d ] km\n\n\n\n", reservation_info[id].total_distance);


	// �װ��� ���� ���� ���
	printf("  (2) [�װ��� ���� ����]�� �Ʒ��� �����ϴ�.\n\n");
	if (reservation_info[id].fly_ver == 1)			printf("        �� [Economy]������� ����Ϸ�.\n");
	else if (reservation_info[id].fly_ver == 2)		printf("        �ڡ� [Business]������� ����Ϸ�.\n");
	else											printf("        �ڡڡ� [First]������� ����Ϸ�.\n");

	printf("\n            [�����]           [������]                [����]\n");
	printf("     ========================================================================\n");

	for (temp = head; temp != NULL; temp = temp->next) {
		printf("          [ %d ]�� ����   ��   [ %d ]�� ����        [ %d ] KRW(��)\n", temp->cur_city, temp->next_city, temp->flight_cost);
	}
	printf("     ========================================================================\n");


	// ��ũ�� ����Ʈ ���
	printf("\n  (3) ���ú� [��¥ �й�]�� �Ʒ��� �����ϴ�. \n\n");
	printLinkedList(head);


	// ȣ�� ���� ���
	printf("\n\n  (4) [ȣ�� ���� ����]�� �Ʒ��� ���� �����߽��ϴ�.\n\n");
	if (reservation_info[id].hotel_ver == 1)				printf("        �� [Cheapest] ������� ȣ�� ���� �Ϸ�\n");
	else if (reservation_info[id].hotel_ver == 2)			printf("        �ڡ� [Resonable] ������� ȣ�� ���� �Ϸ�\n");
	else													printf("        �ڡڡ� [Flex] ������� ȣ�� ���� �Ϸ�\n");
	printf("\n            [����]        [���ù�ȣ]               [ȣ�ڰ���]\n");
	printf("     ================================================================================\n");
	int hot_cnt = 1;
	for (temp = head; temp != NULL; temp = temp->next) {		// �湮�� ���� ���鼭
		for (int i = 0; i < temp->stay; i++) {						// ���� �� �� ���鼭
			printf("         - %d����  :  [ %d ]������ Hotel      [ %d ] KRW(��) �� ����Ϸ�.\n", hot_cnt, temp->cur_city, temp->hotel[i]);
			hot_cnt++;
		}
		printf("\n\n");
	}
	printf("     ================================================================================\n");
}


int main() {
	//--------------------------------------------------------------------------------------------------------//
	//--------------------------[1�ܰ�] ����(100,300)�׷��� �����ϱ� -------------------------------------------//
	//--------------------------------------------------------------------------------------------------------//
	srand(time(NULL));  // ���� �߻��� �ʱ�ȭ

	int** adjMatrix = createAdjMatrix();

	// ���� �׷����� �� ������ �׷��� ����
	Make_Coneected_Graph(adjMatrix);
	// �������� ī�� �س���
	int originMatrix[NODES][NODES];
	for (int i = 0; i < NODES; i++) {
		for (int j = 0; j < NODES; j++) {
			originMatrix[i][j] = adjMatrix[i][j];
		}
	}


	floyd(adjMatrix);       // �÷��̵� ���� : ���� �̵��� �ִܰŸ��� ��� ����

	// printOriginMatrix(originMatrix);  // ���� ��� ���
	// printAdjMatrix(adjMatrix);  // �÷��̵� ��(����) ��� ���



	//--------------------------------------------------------------------------------------------------------//
	//--------------------------[2�ܰ�] ȣ�� RBƮ�� + ������RBƮ�� �����ϱ� -------------------------------------//
	//--------------------------------------------------------------------------------------------------------//
	struct rbtree* HT[NODES];  // 100�� �� ���ø��� ȣ�� ��Ʈ�� �����ϴ� �迭 ����
	// ȣ�� RB tree ����
	Make_Hotel_RBtrees(HT);

	int avail_room[NODES];		// ���ú� ȣ�� ���� ����
	for (int i = 0; i < NODES; i++)	avail_room[i] = HOTELS;	// �ʱ�ȭ

	// ������ RBƮ�� ����
	// RB tree ����
	struct rbtree* BOOKT = newRBtree();
	int booked_num = 0;		// ���� �Ϸ�� ��(BOOKT�� ��尳��) �ʱ�ȭ

	//--------------------------------------------------------------------------------------------------------//
	//--------------------------[3�ܰ�] UI ������ �ϱ�---------------------------------------------------------//
	//--------------------------------------------------------------------------------------------------------//
	// input(����, ����Ⱓ, ��������, �����, �������(cheap/resonable/flex))
	// 
	// - ������ : ����� - 1 - 3 - 6 - 4 - ����� ��θ� �ּ����� �Ÿ��� ������� §��.
	//				��  ���� ��� 1000���� �������� ��� ����(�ؼڰ�)
	// 
	// - ���� : (��, ����, ���) ���ÿ� ���� ȣ�ڰ��ݰ� �װ���(economy, buissniss, first)�� ����.
	//
	// - ����Ⱓ : ��� ���ú��� �յ��ϰ� �й�. ���� ���ڰ� ���� ��� ��������� ������� �Ϸ羿 �� �й�.


	// Output : customer-id, itinerary (timed information of hotels, transportations, sites, dates, total price)
	//
	// - customer-id : �����ڹ�ȣ�� �ο�(�ߺ�x). ������ RBƮ���� ������ ��ȣ�� key�� INSERT.(���� Ȯ��!)
	//
	// - sites : �湮 ���� ���� ����
	// 
	// - dates : ��¥�� �湮���� ����
	// 
	// - trasnportation : ���� ��ο� ���� �װ��� �̵����, ������ ����
	//
	// - timed information of hotels : ���� ���ں� ȣ�� ��������(�����̸�, ȣ���̸�, ��¥��, ����, �Խ�, ��� �ð�)
	// 
	// - total price : �װ��� + ȣ�ں� = �Ѱ����� ����
	//



	// ������ ��� ¥��, ����Ⱓ �й� ���� ¥����!
	// 3-1. INPUT �ޱ� (budget, destination, tour-period ...)
	int cmd, cmd2;		// ��� ��ȣ
	int id;			// �����ڹ�ȣ

	// UI ����!!
	while (true) {
		cout << "\n\n\n\n\n\n\n\n =====================================================================================================================\n";
		cout << "  ����(travel) �ڵ� ���� �ý��� [Ȩ]�Դϴ�. �ɼ��� �������ּ���.\n\n  (1.���ο� ����  /  2.���� ����  /  3.��������Ȯ��  /  " 
			<< "4.�����ڹ�ȣ����  /  5.ȣ������  /  6.�װ�������  /  7.AI ê�� ����  /  8.����) \n\n      - ����(��ȣ)? : ";
		scanf("%d", &cmd);
		if (cmd == 8) {
			printf("\n\n  8. ���Ḧ �����ϼ̽��ϴ�.\n");
			break;
		}
		else if (cmd == 7) {
			cout << "\n\n  7. [���� ��õ AI ê�� ����]�� �����ϰڽ��ϴ�.\n  ����� �����Ӱ� �������ּ���. ���Ḧ ���Ͻø� [exit] �� �Է����ּ���.\n\n";
			using json = nlohmann::json;
			openai::start();

			string previousUserContent = "";
			string previousGPTResponse = "";

			while (true) {
				cout << "Q : ";
				string userContent;
				//cin.ignore();
				getline(cin, userContent);
				if (userContent == "exit") {
					cout << "\n  [���� ��õ AI ê�� ����]�� �����մϴ�. �̿����ּż� �����մϴ�.\n  [Ȩ]���� ���ư��ϴ�.\n";
					break;
				}

				// ���� ��ȭ�� ���� ��ȭ�� �����Ͽ� JSON ��ü ����
				json messages = json::array();
				messages.push_back({ {"role", "user"}, {"content", previousUserContent} });
				messages.push_back({ {"role", "assistant"}, {"content", previousGPTResponse} });
				messages.push_back({ {"role", "user"}, {"content", userContent} });

				json jsonRequest = {
					{"model", "gpt-3.5-turbo"},
					{"messages", messages},
					{"max_tokens", 500},
					{"temperature", 0}
				};

				// Chat API ȣ��
				auto chat = openai::chat().create(jsonRequest);

				// GPT ���� ����
				previousUserContent = userContent;
				previousGPTResponse = chat["choices"][0]["message"]["content"].get<string>();

				// ���� ���
				cout << "GPT : " << previousGPTResponse << '\n';
			}
		}
		else if (cmd == 6) {
			while (true) {
				printf("\n\n  6. [�װ��� ����]�� �˷��帮�ڽ��ϴ�.\n\n  (1.���ú� ���� ����  /  2.�������� ������  /  3.������\n  ����(��ȣ)? : ");
				scanf("%d", &cmd2);
				if (cmd2 == 1) {
					printf("\n\n  6-1. ���ú� ���� ������ �����ص帮�ڽ��ϴ�.(2���� array ����, - : ����X)\n\n");
					printOriginMatrix(originMatrix);
				}
				else if (cmd2 == 2) {
					printf("\n\n  6-1. �������� ���� ������ �����ص帮�ڽ��ϴ�.(2���� array ����)\n\n");
					printAdjMatrix(adjMatrix);
				}
				else if (cmd2 == 3) {
					printf("\n\n  6-3. [Ȩ]���� ���ư��ϴ�.\n");
					break;
				}
				else {
					printf("\n\n  6-3. ���ù�ȣ ����. �ٽ� �������ּ���.\n\n");
				}
			}

		}
		else if (cmd == 5) {
			while (true) {
				printf("\n\n  5. [ȣ�� ����]�� �˷��帮�ڽ��ϴ�.\n\n  (1.Ư������ ȣ�ڰ���  /  2.��絵�� ȣ�ڰ���  /  3.������\n  ����(��ȣ)? : ");
				scanf("%d", &cmd2);
				if (cmd2 == 1) {
					int hot_tmp;
					printf("\n\n  5-1. Ư�������� ȣ�ڰ����� �����ص帮�ڽ��ϴ�.\n  ���ù�ȣ? : ");
					scanf("%d", &hot_tmp);
					printf("\n\n\n\n  %d�� ������ ���� ������ ȣ���� [ %d ]�� �����ֽ��ϴ�.\n  ������ �Ʒ��� �����ϴ�.\n", hot_tmp, avail_room[hot_tmp]);
					printf("  ��� �� ����  --->   RED : ��          BLACK : ��\n");
					printf("\n  %d�� ������ ȣ�� ���� �����Դϴ�. \n", hot_tmp);
					PRINT_BST(HT[hot_tmp]->root, NULL, false);
				}
				else if (cmd2 == 2) {
					printf("\n\n  5-2. ��絵���� ȣ�ڰ����� �����ص帮�ڽ��ϴ�.\n");
					for (int i = 0; i < NODES; i++) {
						printf("\n\n   %d�� ���ô� [ %d ]���� ȣ���� �����ֽ��ϴ�.\n", i, avail_room[i]);
						PRINT_BST(HT[i]->root, NULL, false);
					}
					printf("  ��� �� ����  --->   RED : ��          BLACK : ��\n");
				}
				else if (cmd2 == 3) {
					printf("\n\n  5-3. [Ȩ]���� ���ư��ϴ�.\n");
					break;
				}
				else {
					printf("\n\n  5-3. ���ù�ȣ ����. �ٽ� �������ּ���.\n\n");
				}
			}
		}
		else if (cmd == 4) {
			printf("\n\n  4. [������ ��Ȳ](������ȣ)�� �����帮�ڽ��ϴ�.\n");
			printf("  ��� �� ����  --->   RED : ��          BLACK : ��\n");
			// ������ RBƮ�� ������ֱ�
			PRINT_BST(BOOKT->root, NULL, false);
			printf("\n  ���� ���� �� [ %d ]���� ������ �ּ̽��ϴ�.\n\n  �ƹ� ���ڸ� �Է��ϸ� [Ȩ]���� ���ư��ϴ�.\n  ���� : ", booked_num);
			int tmp;  scanf("%d", &tmp);
		}
		else if (cmd == 3) {
			printf("\n\n  3. [�����Ͻ� ������ Ȯ��]�ص帮�ڽ��ϴ�.\n\n  ������ ������ȣ�� �Է����ּ���.\n\n  customer-id : ");
			scanf("%d", &id);

			// ������ RBƮ������ ã�ƺ���, �����ϸ� ���� ���.
			if (rbtree_find(BOOKT, id) == NULL) {
				printf("\n\n  [������ȣ ����] �Է��Ͻ� ������ ������ȣ( %d ) ������ �������� �ʽ��ϴ�.\n  ������ ������ȣ�� �ٽ� Ȯ�����ּ���.\n\n", id);
				printf("  �ƹ� Ű�� �Է��ϸ�[Ȩ]���� ���ư��ϴ�.\n  �ƹ� Ű : ");
				char tmp[10];  scanf("%s", tmp);
				continue;
			}

			// id�� �����ϹǷ� struct���� ���� ���� ���
			Show_reservation(id);
		}
		else if (cmd == 2) {
			printf("\n\n  2. [���� ����]�� �����ϰڽ��ϴ�.\n\n  �����Ͻ� ������ ������ȣ�� �Է����ּ���.\n\n  customer-id : ");
			scanf("%d", &id);

			// ������ RBƮ������ ã�ƺ���, ��������(����� ȣ�ڵ� ���� �ٽ� ����)
			if (rbtree_find(BOOKT, id) == NULL) {
				printf("\n\n  [������ȣ ����] �Է��Ͻ� ������ ������ȣ( %d ) ������ �������� �ʽ��ϴ�.\n  ������ ������ȣ�� �ٽ� Ȯ�����ּ���.\n\n", id);
				printf("  �ƹ� Ű�� �Է��ϸ�[Ȩ]���� ���ư��ϴ�.\n  �ƹ� Ű : ");
				char tmp[10];  scanf("%s", tmp);
				continue;
			}

			// ���� ���� �ϴ� �����ֱ�
			Show_reservation(id);
			printf("\n\n  �� ���� ���� ���������� �����մϴ�. ������ �����Ͻðڽ��ϱ�?(y/n)\n");
			printf("  �Է�?(y/n) : ");
			char yn[10];	scanf("%s", yn);
			if (yn[0] == 'n' || yn[0] == 'N') {
				printf("\n\n  �������� �ʽ��ϴ�. �ƹ�Ű�� �����ø� [Ȩ]���� ���ư��ϴ�.\n  �ƹ� Ű : ");
				char tmp[10];	scanf("%s", tmp);
				continue;
			}


			printf("\n\n  ������ �����ϰڽ��ϴ�.....\n");
			// id ������ �����ϹǷ� ��� ����.
			// ȣ�� ��� ����.
			linkNode* head = reservation_info[id].head;
			linkNode* temp = head;
			for (temp = head; temp != NULL; temp = temp->next) {
				for (int i = 0; i < temp->stay; i++) {
					RB_INSERT(HT[temp->cur_city], temp->hotel[i]);		// ȣ�� rbƮ���� �ǵ����ֱ�
				}
				// ȣ�� ��밡�� ����� �ǵ�����
				avail_room[temp->cur_city] += temp->stay;
			}

			// ������ id ��� ����
			RB_DELETE(BOOKT, id);
			// ������ �� �� ����
			booked_num -= 1;

			// ������ ����ü�� û��
			strcpy(reservation_info[id].my_name, "");
			reservation_info[id].year = 0;
			reservation_info[id].month = 0;
			reservation_info[id].day = 0;
			reservation_info[id].period = 0;
			reservation_info[id].budget = 0;
			reservation_info[id].total_price = 0;
			reservation_info[id].total_distance = 0;
			reservation_info[id].fly_ver = 0;
			reservation_info[id].hotel_ver = 0;
			reservation_info[id].head = NULL;

			printf("  - �����ȣ ( %d )�� ��� ������ �����Ǿ����ϴ�. �����մϴ�.\n", id);

		}
		else if (cmd == 1) {

			int budget, num_dest, period, start;
			int input_dest[100 + 5];
			int fly_ver, hotel_ver;
			int year, month, day;
			char my_name[100];
			int total_price = 0;		// �� ���

			while (true) {
				printf("\n\n\n  1. ȯ���մϴ�. [���ο� ����]�� �����ϰڽ��ϴ�.\n");

				printf("\n  '����'�� �Է��ϼ���.\n      - ����(����) : ");
				scanf("%d", &budget);

				printf("\n  '�������� ����'�� ���� �Է��ϼ���. (<=100)\n      - ������ ���� : ");
				scanf("%d", &num_dest);
				if (num_dest > 100 || num_dest < 1) {
					printf("\n  [����] : ������ ������ �߸��Ǿ����ϴ�. (1��~100���� �Է��ؾ���)\n");
					exit(0);
				}
				printf("\n\n  '��������'(���ù�ȣ)�� �Է����ּ���. (���ù�ȣ 0~99��)\n      - ��������(��ȣ) : ");
				for (int i = 0; i < num_dest; i++) {
					scanf("%d", &input_dest[i]);
					if (input_dest[i] < 0 || input_dest[i] > 99) {
						printf("\n  [����] : ���ù�ȣ�� �߸��Ǿ����ϴ�. �ٽ� �Է����ּ���. (0~99������ �Է��ؾ���)\n");
						exit(0);
					}
				}
				printf("\n\n  �������� �� '���� ����'�� �Է����ּ���.\n      - ���۵���(��ȣ) : ");
				scanf("%d", &start);
				if (start < 0 || start > 99) {
					printf("\n  [����] : ���۵��ù�ȣ�� �߸��Ǿ����ϴ�. (0~99������ �Է��ؾ���)\n");
					exit(0);
				}
				// �Է��� �������߿� �������� ������ ����
				bool start_flag = false;
				for (int i = 0; i < num_dest; i++) {
					if (input_dest[i] == start)
						start_flag = true;
				}
				if (!start_flag) {
					printf("\n  [����] : ���۵��ð� ���������߿� �������� �ʽ��ϴ�.\n");
					exit(0);
				}

				// input_dest�� 0��° ���ð� ���۵��ð� �ǵ��� ����
				for (int i = 0; i < num_dest; i++) {
					if (input_dest[i] == start) {
						int tmp = input_dest[0];
						input_dest[0] = input_dest[i];
						input_dest[i] = tmp;
					}
				}



				printf("\n\n  ���� �Ⱓ�� �Է��ϼ���.\n      - ���� �Ⱓ(��) : ");
				scanf("%d", &period);

				// �װ���, ȣ�� ���� �ɼ� ����
				while (true) {
					printf("\n\n  �װ��� �ɼ��� �����ϰڽ��ϴ�. [�װ���]�� ��� �����ұ��? (1.Economy,  2.Business,  3.First)\n        - �Է� ����(�ɼǹ�ȣ) : ");
					scanf("%d", &fly_ver);
					if (fly_ver == 1 || fly_ver == 2 || fly_ver == 3)	break;
					else												printf("          [�װ��� �ɼ� ����] : ���ڸ� �߸� �Է��ϼ̽��ϴ�. 1,2,3���߿��� �ٽ� �Է����ּ���.\n");
				}
				while (true) {
					printf("\n\n  �������� ȣ�� �ɼ��� �����ϰڽ��ϴ�. [ȣ��]�� ��� �����ұ��? (1.Cheapest,  2.Reasonable,  3.Flex)\n        - �Է� ����(�ɼǹ�ȣ) : ");
					scanf("%d", &hotel_ver);
					if (hotel_ver == 1 || hotel_ver == 2 || hotel_ver == 3)	break;
					else												printf("          [ȣ�� �ɼ� ����] : ���ڸ� �߸� �Է��ϼ̽��ϴ�. 1,2,3���߿��� �ٽ� �Է����ּ���.\n");
				}

				// ��� ��¥ �Է�
				printf("\n\n  [��� ��¥]�� �˷��ּ���.  - �⵵(year) : ");
				scanf("%d", &year);
				printf("                            -  ��(month) : ");
				scanf("%d", &month);
				printf("                            -  ��(day) : ");
				scanf("%d", &day);

				// ������ �̸� �ޱ�
				printf("\n\n  ���������� [������ �̸�]�� �Է����ּ���.\n        - ������ �̸�(��������) : ");
				scanf("%s", my_name);


				printf("\n\n\n  ������ �������� �Է����ּż� �����մϴ�. �Է��Ͻ� ������� �ڵ� ������ �����ұ��? (y/n)\n  �Է�(y/n) : ");
				char yn[10];
				scanf("%s", yn);
				if (yn[0] == 'y' || yn[0] == 'Y')
					break;
				else
					printf("  �ٽ� �����ϱ⸦ �����ϼ̽��ϴ�. ���� ó������ ���ư��ϴ�.\n\n");
			}	// input�ޱ� ��

			// 3-2. ������ ���(�湮����)¥��
			int* sorted_dest = make_dest_order(num_dest, input_dest, start, adjMatrix);
			int total_distance = Sum_distance_dest_order(sorted_dest, num_dest, adjMatrix);

			// ��� ����غ���
			printf("\n\n\n �ѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤ�[AI���� �����]�ѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤ�\n");
			printf("\n\n  (1) �ּҺ�� �װ� �̵� ��� : ");
			for (int i = 0; i < num_dest; i++)
				printf(" [%d]���� -> ", sorted_dest[i]);
			printf(" [%d]����\n\n", start);
			printf("        - �� �̵��Ÿ� : [ %d ] km\n\n\n\n", total_distance);


			// 3-3. �̵� ���ú� ��ũ�� ����Ʈ �����
			linkNode* ct_head = NULL;
			linkNode* temp = ct_head;

			// ��� �߰� & �װ��� ���� ���� ���
			printf("  (2) [�װ��� ���� ����]�� �Ʒ��� �����ϴ�.\n\n");
			if (fly_ver == 1)			printf("        �� [Economy]������� ����Ϸ�.\n");
			else if (fly_ver == 2)		printf("        �ڡ� [Business]������� ����Ϸ�.\n");
			else						printf("        �ڡڡ� [First]������� ����Ϸ�.\n");

			printf("\n            [�����]           [������]                [����]\n");
			printf("     ========================================================================\n");

			for (int i = 0; i < num_dest; i++) {
				// ���絵��, ��������, �ӹ��³���, �װ���, �װ�����, ȣ�ڿ����� ����
				temp = addNode(&ct_head, sorted_dest, period / num_dest, i, num_dest, fly_ver, adjMatrix);
				printf("          [ %d ]�� ����   ��   [ %d ]�� ����        [ %d ] KRW(��)\n", temp->cur_city, temp->next_city, temp->flight_cost);
				total_price += temp->flight_cost;				// �� ���� ����
			}
			printf("     ========================================================================\n");
			// ����Ⱓ �й��ϱ�(�������� ������ �տ������� �Ϸ羿 ����)
			int rest_day = period % num_dest;
			if (rest_day > 0) {
				temp = ct_head;
				while (temp != NULL) {
					temp->stay += 1;
					rest_day--;
					if (rest_day == 0)
						break;		// �� ���������� ����
					temp = temp->next;
				}
			}



			// ��ũ�� ����Ʈ ���
			printf("\n  (3) ���ú� [��¥ �й�]�� �Ʒ��� ���� �����մϴ�. \n\n");
			printLinkedList(ct_head);


			// 3-4. ȣ�� �����ϱ� (���ú� ����Ⱓ�� ����)
			// �����ڰ� 900���̻� : ����
			if (booked_num >= 900) {
				printf("\n  [������ ����] �˼��մϴ�. ���� �Ϸ��ڰ� 900���� �Ѿ ���̻� ������ �Ұ����մϴ�.\n  ������ �ı�˴ϴ�....\n\n");
				printf("  �ƹ� ���ڳ� ������ [Ȩ]���� ���ư��ϴ�.\n  �ƹ� ���� : ");
				int tmp;  scanf("%d", &tmp);
				continue;
			}
			// �� ���õ� �ѷ����鼭
			bool room_flag = true;
			for (temp = ct_head; temp != NULL; temp = temp->next) {
				// �� ���ÿ� ȣ�ڼ��� �����ϴٸ�(�Ϸ翡 ȣ�� �ϳ��� ������)
				if (temp->stay > avail_room[temp->cur_city]) {
					room_flag = false;
					break;
				}
			}
			// ȣ�ڼ��� �����ϹǷ� ���� �ı�
			if (!room_flag) {
				printf("\n  [ȣ�� ����] �˼��մϴ�. %d�� ���ÿ� ���డ���� ȣ���� ���ġ �ʽ��ϴ�.\n  �ٸ� ���ø� �����ϰų� �������ڸ� �ٿ��ּ���.\n  ������ �ı�˴ϴ�....\n\n", temp->cur_city);
				printf("  �ƹ� ���ڳ� ������ [Ȩ]���� ���ư��ϴ�.\n  �ƹ� ���� : ");
				int tmp;  scanf("%d", &tmp);
				continue;
			}

			// ȣ�� ���� ����
			printf("\n\n  (4) [ȣ�� ���� ����]�� �Ʒ��� ���� �����߽��ϴ�.\n\n");
			if (hotel_ver == 1)				printf("        �� [Cheapest] ������� ȣ�� ���� �Ϸ�\n");
			else if (hotel_ver == 2)		printf("        �ڡ� [Resonable] ������� ȣ�� ���� �Ϸ�\n");
			else							printf("        �ڡڡ� [Flex] ������� ȣ�� ���� �Ϸ�\n");
			printf("\n            [����]        [���ù�ȣ]               [ȣ�ڰ���]\n");
			printf("     ================================================================================\n");
			int hot_cnt = 1;
			for (temp = ct_head; temp != NULL; temp = temp->next) {		// �湮�� ���� ���鼭
				for (int i = 0; i < temp->stay; i++) {						// ���� �� �� ���鼭
					// cheapest, resonable, flex�ɼ� ���
					struct node_t* booking_hotel;
					if (hotel_ver == 1)				booking_hotel = MinKeyOfRBtree(HT[temp->cur_city]);		// 1) cheapest (left most)
					else if (hotel_ver == 2)		booking_hotel = HT[temp->cur_city]->root;				// 2) resonable (root)
					else							booking_hotel = MaxKeyOfRBtree(HT[temp->cur_city]);		// 3) flex (right most)


					if (booking_hotel) {
						// ȣ�ڳ������� ����(����Ȯ��)
						int booking_price = booking_hotel->key;
						// printf("  ���� : %d,  ������� key���� : %d\n", temp->city_num, cheap_hotel->key);
						RB_DELETE(HT[temp->cur_city], booking_hotel->key);
						avail_room[temp->cur_city] -= 1;	// ���డ���� ȣ�� �� 1�� ����
						temp->hotel[i] = booking_price;		// ���� ȣ�� ����(key) ���
						printf("         - %d����  :  [ %d ]������ Hotel      [ %d ] KRW(��) �� ����Ϸ�.\n", hot_cnt, temp->cur_city, booking_price);
						hot_cnt++;
						total_price += booking_price;			// �� ��� ����
					}
				}
				printf("\n\n");
			}
			printf("     ================================================================================\n");



			// Output : customer-id, itinerary (timed information of hotels, transportations, sites, dates, total price)
			//
			// - customer-id : �����ڹ�ȣ�� �ο�(�ߺ�x). ������ RBƮ���� ������ ��ȣ�� key�� INSERT.(���� Ȯ��!)
			//
			// - sites : �湮 ���� ���� ����
			// 
			// - dates : ��¥�� �湮���� ����
			// 
			// - trasnportation : ���� ��ο� ���� �װ��� �̵����, ������ ����
			//
			// - timed information of hotels : ���� ���ں� ȣ�� ��������(�����̸�, ȣ���̸�, ��¥��, ����, �Խ�, ��� �ð�)
			// 
			// - total price : �װ��� + ȣ�ں� = �Ѱ����� ����
			// 




			// struct�� ��� ���������� �� ������ �� �� ����.
			// ���� ����Ʈ ���� �ٽ� �״�� ���� �� �ֵ���. (list_head , total_distance , hotel_ver, fly_ver, customer-id, name, start_date)�� struct�� ��Ҹ� ����.
			//													�� list_head�� �����̵����, �װ��ǰ���, ȣ�ڰ���, �ӹ������� ��ϵǾ� �ִ�.

			// rbƮ�� ��忡 �߰��ϴ°� ����. �׳� struct[100][100]¥�� ������.
			// ���� üũ�ϰ� ���꿡 ������ ���������ؾ���.
			// �ϴ� 1.����üũ 2.����Ȯ�� 3.������� �����ϰ� ��������.

			printf("\n\n    �� �� ���� : %d (KRW)\n", budget);
			printf("    �� �� ��� : %d (KRW)\n\n", total_price);
			// ������ ���ڶ�� ȣ�� ���� ���� �����ϰ� �����ı�
			if (budget < total_price) {
				// ȣ�� ��� ����.
				for (temp = ct_head; temp != NULL; temp = temp->next) {
					for (int i = 0; i < temp->stay; i++) {
						RB_INSERT(HT[temp->cur_city], temp->hotel[i]);		// ȣ�� rbƮ���� �ǵ����ֱ�
					}
					// ȣ�� ��밡�� ����� �ǵ�����
					avail_room[temp->cur_city] += temp->stay;
				}
				printf("\n  [���� ����] �˼��մϴ�. ���꺸�� �� �������� �� �����Ƿ� ������ �Ұ����մϴ�.\n  ���� : %d (KRW)\n  ���� �Ѻ�� : %d (KRW)\n  ������ �÷��ֽðų� ȣ��, ���� �ɼ��� ���缭 �ٽ� �������ּ���.\n  ������ �ı�˴ϴ�....\n\n", budget, total_price);
				printf("  �ƹ� ���ڳ� ������ [Ȩ]���� ���ư��ϴ�.\n  �ƹ� ���� : ");
				int tmp;  scanf("%d", &tmp);
				continue;
			}

			// 3-5. customer-id ���� & ������RBƮ���� ����
			int random_customer_id = 0;
			do {
				random_customer_id = (rand() % 900) + 100;		// 100 ~ 999���� �����ȣ
			} while (rbtree_find(BOOKT, random_customer_id) != NULL);

			// customer-id�� ������RB tree�� ����
			RB_INSERT(BOOKT, random_customer_id);
			booked_num += 1;		// ������ 1�� ����

			// 3-6. �������� �����ؼ� struct�� ���
			strcpy(reservation_info[random_customer_id].my_name, my_name);
			reservation_info[random_customer_id].year = year;
			reservation_info[random_customer_id].month = month;
			reservation_info[random_customer_id].day = day;
			reservation_info[random_customer_id].period = period;
			reservation_info[random_customer_id].budget = budget;
			reservation_info[random_customer_id].total_price = total_price;
			reservation_info[random_customer_id].total_distance = total_distance;
			reservation_info[random_customer_id].fly_ver = fly_ver;
			reservation_info[random_customer_id].hotel_ver = hotel_ver;
			reservation_info[random_customer_id].head = ct_head;



			printf("\n �ѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤ�\n");
			printf("\n\n  ���� ���� �װ��ǰ� ȣ�� ������ �Ϸ�Ǿ����ϴ�.\n\n  ������ �����ȣ��  (customer-id) : [ %d ] �Դϴ�.\n\n  �����մϴ�. �ƹ� ���ڸ� �Է��ϸ� [Ȩ]���� ���ư��ϴ�.\n  �ƹ� ���� : ", random_customer_id);
			int tmp;  scanf("%d", &tmp);

		}// 1.���ο� ���� ��
		else {
			printf("\n\n  [��ɾ� ����] : ��ɾ �߸� �Է��ϼ̽��ϴ�. �ٽ� �Է����ּ���.\n\n");
			printf("  �ƹ� ���ڳ� ������ [Ȩ]���� ���ư��ϴ�.\n  �ƹ� ���� : ");
			int tmp;  scanf("%d", &tmp);
		}
	}// ��ɾ� while�� ��

	//--------------------------------------------------------------------------------------------------------//
	//--------------------------[������ �ܰ�] �޸� �����ϱ� --------------------------------------------------//
	//--------------------------------------------------------------------------------------------------------//
	// ������ ȣ�ڷ�Ʈ���� �޸𸮿��� ����
	printf("\n  ����ߴ� '�޸𸮸� ������'�Դϴ�. ��ø� ��ٷ��ּ���....\n\n");
	for (int i = 0; i < 100; i++) {
		free(HT[i]);
	}
	printf("\n  '����'�ϰڽ��ϴ�. �ٽ� �̿����ּ���. �����մϴ�.\n");
	freeAdjMatrix(adjMatrix);   // ������� �޸� ����

	return 0;
}








