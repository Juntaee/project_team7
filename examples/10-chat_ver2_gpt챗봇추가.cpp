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

const int INF = 987654321;     // INF값 9억.


/*
   [05.17]
   1. 그래프(100,300)을 생성 + 간선 중복없이 300개 랜덤생성 + 연결그래프 확인까지 완료
   - 인접행렬 100x100이므로 memory : 40kb

   [05.18]
   1. 간선에 랜덤 가중치(1~9)를 줬음. 도시간 거리 distance를 의미.
   2. 플로이드 워셜 알고리즘으로 도시간 이동할 때, 항상 최단거리로 이동하도록 인접행렬을 업데이트 했음.
   3. RB트리의 기본 함수들을 report2에서 그대로 가져왔음.
   4. 호텔RB트리를 만들자. (key : 호텔 가격)
 */
#define NODES (100)     // 도시 개수 : 100개
#define EDGES (300)     // transportation 개수(항공선 개수) : 300개
#define HOTELS (100)     // 각 도시당 호텔 개수 : 100개씩 (총 10,000개)




//ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ RB트리 함수들 시작 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ//

enum Color {
        RED = 0,
        BLACK = 1
};

// 노드 구조체
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

// leaf = nil 노드
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

// rand()값 중복되면 true 리턴
bool is_duplicate(int* list, int value, int end) {

        for (int i = 0; i < end; i++) {
                if (list[i] == value) {
                        return true;
                }
        }
        return false;
}

// 노드 삭제
void del_node(struct node_t* n) {
        if (n != NULL) {
                del_node(n->left);
                del_node(n->right);
        }
        free(n);
}

// rb트리 전체 초기화
void delete_rbtree(struct rbtree* tree) {
        del_node(tree->root);
        free(tree);
}

// 새로운 노드 생성
struct node_t* newNode(struct rbtree* tree, int k) {
        struct node_t* new_node = (struct node_t*)malloc(sizeof(struct node_t));

        new_node->parent = NULL;
        new_node->left = tree->NIL;
        new_node->right = tree->NIL;
        new_node->key = k;
        new_node->color = RED;

        return new_node;
}

// rb트리 시작 (루트 메모리 할당)
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

// RB트리에서 최소 key 찾기 : cheapest hotel reserve
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

// RB트리에서 최대 key 찾기 : flex hotel reserve
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

// SUCCESSOR 찾기
struct node_t* tree_successor(struct rbtree* t, struct node_t* n) {
        // right child가 존재할 경우, 우측 노드의 제일 왼쪽을 찾으면 됨
        // right sub tree의 최솟값
        if (n->right != NULL) {
                // temp가 NULL이 될때까지 왼쪽으로 이동
                struct node_t* temp_c = n->right;
                while (temp_c->left != NULL) {
                        temp_c = temp_c->left;
                }
                return temp_c;
        }
        // right node가 없으면, n의 parent 선언 후
        struct node_t* temp_p = n->parent;
        // 위로 올라가면서 temp node가 left child가 될때까지 반복
        // 왼쪽위로 쭉 올라가다가 오른쪽위로 처음으로 올라갔을 때 그 놈이 successor이다.
        while (temp_p != NULL && n == temp_p->right) {
                n = temp_p;
                temp_p = temp_p->parent;
        }

        return temp_p;
}

// PREDECCESSOR 찾기
struct node_t* tree_predeccessor(struct rbtree* t, struct node_t* n) {
        // left child가 존재할 경우, 좌측 노드의 제일 오른쪽을 찾으면 됨
        // Left sub tree의 최댓값
        if (n->left != t->NIL) {
                // temp가 NULL이 될때까지 왼쪽으로 이동
                struct node_t* temp_c = n->left;
                while (temp_c->right != t->NIL) {
                        temp_c = temp_c->right;
                }
                return temp_c;
        }
        // left node가 없으면, n의 parent 선언 후
        struct node_t* temp_p = n->parent;
        // 위로 올라가면서 temp node가 right child가 될때까지 반복
        // 오른쪽 쭉 타고올라가다가 처음으로 왼쪽으로 올라갔을 때, 그놈이 PREDECCESSOR이다.
        while (temp_p != NULL && n == temp_p->left) {
                n = temp_p;
                temp_p = temp_p->parent;
        }

        return temp_p;
}

// 왼쪽 회전
void right_rotation(struct rbtree* tree, struct node_t* x) {
        // TODO!

        struct node_t* y;

        // 1. target의 left으로 y 설정
        y = x->left;
        // 2. y의 오른쪽 서브트리를 target의 왼쪽 서브트리로 옮김
        x->left = y->right;
        // 3. y의 오른쪽 노드가 NIL이 아니라면, y의 오른쪽 노드 부모를 target으로 설정
        if (y->right != tree->NIL) {
                y->right->parent = x;
        }
        // 4. y의 부모 노드를 target의 부모 노드로 설정
        y->parent = x->parent;
        // 5. target의 부모 노드가 nil이라면, 트리 구조체의 root를 y로 설정
        if (x->parent == tree->NIL)
                tree->root = y;
        // 6. target이 target 부모 노드의 왼쪽이면, target 부모의 왼쪽을 y로 설정
        else if (x == x->parent->left)
                x->parent->left = y;
        // 7. target이 target 부모 노드의 오른쪽이면, target 부모의 오른쪽을 y로 설정
        else
                x->parent->right = y;
        // 8. target을 y의 오른쪽으로 설정
        y->right = x;
        // 9. target의 부모를 y로 설정
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

// search' 용 find 버전 (사용자 버전)
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
                // z의 부모가 조부모의 왼쪽 서브 트리일 경우
                if (z->parent == z->parent->parent->left) {
                        y = z->parent->parent->right;

                        // CASE 1 : 노드 z의 삼촌 y가 적색인 경우
                        if (y->color == RED) {
                                z->parent->color = BLACK;
                                y->color = BLACK;
                                z->parent->parent->color = RED;
                                z = z->parent->parent;
                        }
                        // CASE 2 : z의 삼촌 y가 흑색이며의 z가 오른쪽 자식인 경우
                        else {
                                if (z == z->parent->right) {
                                        z = z->parent;
                                        left_rotation(t, z);
                                }
                                // CASE 3 : z의 삼촌 y가 흑색이며의 z가 오른쪽 자식인 경우
                                z->parent->color = BLACK;
                                z->parent->parent->color = RED;
                                right_rotation(t, z->parent->parent);
                        }
                }
                // z의 부모가 조부모의 왼쪽 서브 트리일 경우
                else {
                        y = z->parent->parent->left;

                        // CASE 4 : 노드 z의 삼촌 y가 적색인 경우
                        if (y->color == RED) {
                                z->parent->color = BLACK;
                                y->color = BLACK;
                                z->parent->parent->color = RED;
                                z = z->parent->parent;
                        }
                        // CASE 5 : z의 삼촌 y가 흑색이며의 z가 오른쪽 자식인 경우
                        else {
                                if (z == z->parent->left) {
                                        z = z->parent;
                                        right_rotation(t, z);
                                }
                                // CASE 6 : z의 삼촌 y가 흑색이며의 z가 오른쪽 자식인 경우
                                z->parent->color = BLACK;
                                z->parent->parent->color = RED;
                                left_rotation(t, z->parent->parent);
                        }
                }
        }
        t->root->color = BLACK;
}

// 노드 삽입
void RB_INSERT(struct rbtree* T, int k) {
        // NIL과 root node
        struct node_t* y = T->NIL;
        struct node_t* x = T->root;

        // 새로운 노드 생성
        struct node_t* z = newNode(T, k);



        // BST 트리와 같은 방식으로 노드 삽입
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

        // Fixup 수행
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

                        // CASE 1 : x의 형제 w가 적색인 경우
                        if (w->color == RED) {
                                w->color = BLACK;
                                x->parent->color = RED;
                                left_rotation(t, x->parent);
                                w = x->parent->right;
                        }

                        // CASE 2 : x의 형제 w는 흑색이고 w의 두 지식이 모두 흑색인 경우
                        if (w->left->color == BLACK && w->right->color == BLACK) {
                                w->color = RED;
                                x = x->parent;
                        }

                        // CASE 3 : x의 형제 w는 흑색, w의 왼쪽 자식은 적색, w의 오른쪽 자신은 흑색인 경우
                        else {
                                if (w->right->color == BLACK) {
                                        w->left->color = BLACK;
                                        w->color = RED;
                                        right_rotation(t, w);
                                        w = x->parent->right;
                                }

                                // CASE 4 : x의 형제 w는 흑색이고 w의 오른쪽 자식은 적색인 경우
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

                        // CASE 5 : x의 형제 w가 적색인 경우
                        if (w->color == RED) {
                                w->color = BLACK;
                                x->parent->color = RED;
                                right_rotation(t, x->parent);
                                w = x->parent->left;
                        }

                        // CASE 6 : x의 형제 w는 흑색이고 w의 두 지식이 모두 흑색인 경우
                        if (w->right->color == BLACK && w->left->color == BLACK) {
                                w->color = RED;
                                x = x->parent;
                        }

                        // CASE 7 : x의 형제 w는 흑색, w의 왼쪽 자식은 적색, w의 오른쪽 자신은 흑색인 경우
                        else
                        {
                                if (w->left->color == BLACK) {
                                        w->right->color = BLACK;
                                        w->color = RED;
                                        left_rotation(t, w);
                                        w = x->parent->left;
                                }

                                // CASE 8 : x의 형제 w는 흑색이고 w의 오른쪽 자식은 적색인 경우
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



// 3-4. SEARCH함수 구현하기
// 1. key값이 존재하면 key 리턴
// 2. key값이 없으면 key의 successor 리턴
// 3. successor도 없으면 Nil리턴
struct node_t* SEARCH(struct rbtree* tree, int k) {
        struct node_t* result = RB_find_ver2(tree, k);          // 일단 k 있는지 찾아보고

        if (result->key == k) {         // 1. key값 존재
                return result;
        }
        else if (result->key > k) {             // 2. key값은 없지만 그의 부모가 바로 successor인경우
                return result;
        }
        else if (result->key < k) {             // 2. key값 없고, 부모도 key보다 작으면, 부모의 successor를 리턴
                struct node_t* succ_result = tree_successor(tree, result);
                if (succ_result == NULL)                        return NULL;
                else if (succ_result->key > k)          return succ_result;
                else                                                            return NULL;                    // 3. successor가 k보다 작다면 = 존재x라는 의미 = nil을 리턴
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
        if (n->key >= 0) {      // nil노드 출력하지 않기 위함 : nil.key = -1로 초기화 되어있음
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
                        printf("%s\n", "○");
                }
                else {
                        printf("%s\n", "●");
                }

                if (prev) {
                        prev->str = prev_str;
                }
                trunk_temp->str = "   |";
        }
        PRINT_BST(n->left, trunk_temp, false);
}


//ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ RB트리 함수들 끝 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ//




// 플로이드 워셜을 위해서 인접행렬을 무한대로 초기화
void adj_init_INF(int** adjMatrix)
{
        for (int i = 0; i < NODES; i++)
                for (int j = 0; j < NODES; j++) {
                        adjMatrix[i][j] = INF;
                        if (i == j) adjMatrix[i][j] = 0;
                }
}

// 모든 도시 -> 모든 도시의 최단경로를 알기 위해서 플로이드워셜 알고리즘 사용
// 효과 : 각 도시를 이동할 때에는 항상 최단경로로 이동하게 됨.
void floyd(int** adjMatrix)
{
        for (int m = 0; m < NODES; m++) {       // 경유도시
                for (int s = 0; s < NODES; s++) {       // 시작도시
                        for (int e = 0; e < NODES; e++) {       // 도착도시
                                                                // 중간경로가 불가능하면 건너뜀!
                                if (adjMatrix[s][m] == INF || adjMatrix[m][e] == INF)
                                        continue;
                                //가운데를 거쳐가는 것이 더 빠르면 그걸로 업데이트한다.
                                if (adjMatrix[s][e] > adjMatrix[s][m] + adjMatrix[m][e])
                                        adjMatrix[s][e] = adjMatrix[s][m] + adjMatrix[m][e];
                        }
                }
        }
}

// 인접 행렬을 생성하고 초기화하는 함수
int** createAdjMatrix() {
        int** adjMatrix = (int**)malloc((NODES + 5) * sizeof(int*));
        for (int i = 0; i < NODES; i++) {
                adjMatrix[i] = (int*)malloc((NODES + 5) * sizeof(int));       // 여유롭게 adj[105][105]로 인접행렬 생성
                for (int j = 0; j < NODES; j++) {
                        adjMatrix[i][j] = 0;  // 초기값은 일단 0으로 설정 -> 나중에 INF로 초기화(플로이드)
                }
        }
        return adjMatrix;
}

// 인접 행렬 메모리 해제
void freeAdjMatrix(int** adjMatrix) {
        for (int i = 0; i < NODES + 5; i++) {
                free(adjMatrix[i]);
        }
        free(adjMatrix);
}

// 두 노드 사이에 간선을 추가하는 함수 : 간선의 가중치는 도시 사이의 거리를 의미함.
void addEdge(int** adjMatrix, int city1, int city2, int weight) {
        adjMatrix[city1][city2] = weight;
        adjMatrix[city2][city1] = weight;       // 도시1 ~ 도시2 사이의 거리 입력
}

// DFS (깊이 우선 탐색) 함수 : 연결함수임을 확인
void dfs(int** adjMatrix, bool visited[], int city) {
        visited[city] = true;

        for (int i = 0; i < NODES; i++) {
                if (adjMatrix[city][i] < INF && !visited[i]) {  // INF보다 작은값이면 직접 연결된 도시.
                        dfs(adjMatrix, visited, i);
                }
        }
}

// 그래프가 연결 그래프인지 확인하는 함수
bool isConnectedGraph(int** adjMatrix) {
        bool visited[NODES];
        for (int i = 0; i < NODES; i++) {
                visited[i] = false;
        }

        // DFS (깊이 우선 탐색)
        dfs(adjMatrix, visited, 0); // 0번도시부터 방문 시작

        // 모든 노드를 방문했는지 확인
        for (int i = 0; i < NODES; i++) {
                if (!visited[i]) {
                        return false;
                }
        }

        return true;
}


// 경유포함 최소거리(플로이드 후)정보를 출력하는 함수
void printAdjMatrix(int** adjMatrix) {
        for (int i = 0; i < NODES; i++) {
                for (int j = 0; j < NODES; j++) {
                        (adjMatrix[i][j] == INF) ? printf("- ") : printf("%d ", adjMatrix[i][j]);
                }
                printf("\n");
        }
}

// 직항 정보 출력하는 함수
void printOriginMatrix(int originMatrix[][NODES]) {
        for (int i = 0; i < NODES; i++) {
                for (int j = 0; j < NODES; j++) {
                        (originMatrix[i][j] == INF) ? printf("- ") : printf("%d ", originMatrix[i][j]);
                }
                printf("\n");
        }
}

// 연결 그래프 G(100, 300) 생성
void Make_Coneected_Graph(int** adjMatrix) {
        while (true) {
                // 인접 행렬 INF로 초기화
                adj_init_INF(adjMatrix);

                int edgeCount = 0;
                while (edgeCount < EDGES) {
                        int city1 = rand() % NODES;
                        int city2 = rand() % NODES;     // 항공선 연결할 도시를 임의로 선택(300개의 항공선)
                        int distance = (rand() % 900) + 100;        // 도시간 거리 : 100~999중 랜덤값

                        // 중복(이미 연결?)된 간선인지 확인 및 자체 루프 확인
                        if (city1 != city2 && adjMatrix[city1][city2] == INF && adjMatrix[city2][city1] == INF) {
                                addEdge(adjMatrix, city1, city2, distance);
                                edgeCount++;
                        }
                }

                if (isConnectedGraph(adjMatrix)) {
                        break;  // 연결 그래프가 생성되었으면 반복 종료
                }
        }
}

// 호텔RB트리 생성하기
void Make_Hotel_RBtrees(struct rbtree* HT[NODES]) {
        for (int i = 0; i < NODES; i++) {
                HT[i] = newRBtree();  // 도시마다 호텔 루트를 생성하여 배열에 할당
        }

        // 호텔루트마다 100개씩 호텔 삽입(총 10,000개 삽입) : 적절한 가격이 key
        // 적절한 랜덤 가격 범위 : 10만원 . 10만100원 .....49만9900원 (10만~49만9900원. 100원단위로) + 중복은 제거

        int random_price;
        // identical value 대입을 위한 체크용 리스트
        int* A = (int*)malloc(HOTELS * sizeof(int));

        // 각 도시마다 호텔100개씩 생성
        for (int i = 0; i < NODES; i++) {       // i 도시마다
                for (int j = 0; j < HOTELS; j++)        A[j] = 0;       // 중복 체크 배열 초기화

                // 100개의 랜덤 가격 insert
                for (int j = 0; j < HOTELS; j++) {              // j 호텔
                        do {
                                random_price = (rand() % 40 + 10) * 10000 + (rand() % 10) * 1000 + (rand() % 10) * 100;  // 10만원~49만9900원(1000원 단위)
                        } while (is_duplicate(A, random_price, j));
                        A[j] = random_price;
                        // A[i]에 숫자 대입
                        // key를 RB tree에 삽입
                        RB_INSERT(HT[i], random_price);
                }

                // 호텔 RB tree 출력
                /*
                   printf("\n\n  Random KEY (호텔)값 100개  :  ");
                   for (int i = 0; i < HOTELS; i++)        printf(" %d ", A[i]);
                   printf("\n\n   RB트리 출력!! \n\n");
                   PRINT_BST(HT[i]->root, NULL, false);
                 */


        }
}

// 어레이 요소들 랜덤으로 섞기(0번째 시작점은 고정)
void shuffle_array(int* arr, int size) {

        for (int i = size - 1; i > 0; i--) {
                // 랜덤 인덱스 생성
                int j = (rand() % i) + 1;       // 0번째는 선택할 수 없도록함.

                // 현재 요소와 랜덤요소를 스왑
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
        }
}

// 주어진 방문순서로 방문해보고 거리의 합 리턴
int Sum_distance_dest_order(int* input_dest, int num_dest, int** adjMatrix) {
        int sum = 0;
        for (int i = 0; i < num_dest - 1; i++) {
                sum += adjMatrix[input_dest[i]][input_dest[i + 1]];
        }
        // 끝점->시작점으로 돌아오는 거리 마지막으로 더해주기
        sum += adjMatrix[input_dest[num_dest - 1]][input_dest[0]];

        return sum;
}

// 목적지 경로(방문순서)짜기
int* make_dest_order(int num_dest, int* input_dest, int start, int** adjMatrix) {
        // ERROR CODE
        if (num_dest <= 0 || num_dest > 100) {
                printf("\n\n  ERROR : 목적지의 개수가 잘못되었습니다.\n");
                exit(0);
        }

        // 결과 어레이 생성
        int* sorted_dest = (int*)malloc((num_dest + 5) * sizeof(int));

        if (num_dest <= 1) {    // 1개일때(예외처리)
                sorted_dest[0] = start;
        }
        // 랜덤으로 1000개 후보중 최소로 결정
        else {  // 21~100개 : 랜덤
                int min_cost = INF;
                int random_list[100 + 5];
                // input을 복사
                for (int i = 0; i < num_dest; i++)
                        random_list[i] = input_dest[i];

                // 1000개의 랜덤 경로를 만들어서 최소비교
                for (int i = 0; i < 1000; i++) {
                        shuffle_array(random_list + 1, num_dest - 1);   // 섞고(시작점은 안섞이도록 수정!)
                        int result = Sum_distance_dest_order(random_list, num_dest, adjMatrix);         // 계산해보고
                                                                                                        // 최소면 방문순서 복사
                        if (result < min_cost) {
                                min_cost = result;
                                for (int j = 0; j < num_dest; j++)
                                        sorted_dest[j] = random_list[j];
                        }
                }
        }

        return sorted_dest;
}

// 링크드 리스트의 노드 구조체 정의(노드 = 도시)
typedef struct linkNode {
        int cur_city;           // 도시번호
        int stay;                       // 머무는 날 수
        int hotel[100 + 5];     // 예약된 호텔가격
        int next_city;          // 다음 도시 번호
        int flight_cost;        // 항공권 가격
        struct linkNode* next;
} linkNode;

// 예약자 확인할때 필요한 구조체
typedef struct {
        char my_name[100];                      // 여행자 이름
        int year;                               // 여행시작 날짜
        int month;
        int day;
        int period;                             // 여행 기간
        int budget;                             // 얘산
        int total_price;                        // 총 가격
        int total_distance;             // 총 이동거리
        int fly_ver;                    // 항공권 옵션
        int hotel_ver;                  // 호텔 옵션
        linkNode* head;                 // 도시이동 링크드 정보들
} Reservation_info;
Reservation_info reservation_info[1000];

// 새로운 노드를 생성하는 함수
linkNode* createNode(int cur_city, int stay, int next_city, int flight_cost) {
        linkNode* nNode = (linkNode*)malloc(sizeof(linkNode));
        nNode->cur_city = cur_city;
        nNode->stay = stay;
        nNode->next_city = next_city;
        nNode->flight_cost = flight_cost;
        nNode->next = NULL;
        return nNode;
}

// 링크드 리스트에 노드를 추가하는 함수
linkNode* addNode(linkNode** head, int* sorted_dest, int stay, int idx, int num_dest, int fly_ver, int** adjMatrix) {
        // 다음도시
        int cur_city = sorted_dest[idx];
        int next_city;
        if (idx == num_dest - 1)        next_city = sorted_dest[0];             // 마지막 도시면 처음도시로 회귀!
        else                                    next_city = sorted_dest[idx + 1];

        // 항공권 가격 결정
        int flight_cost;
        int distance = adjMatrix[cur_city][next_city];
        if (fly_ver == 1)                       flight_cost = distance * 1500;          // 이코노미(25%할인)
        else if (fly_ver == 2)          flight_cost = distance * 2000;          // 비즈니스(기준)
        else                                            flight_cost = distance * 2600;          // 퍼스트(30%상향)


        linkNode* nNode = createNode(cur_city, stay, next_city, flight_cost);

        if (*head == NULL) {
                // 리스트가 비어있을 경우
                *head = nNode;
        }
        else {
                // 리스트가 비어있지 않을 경우
                linkNode* temp = *head;
                while (temp->next != NULL) {
                        temp = temp->next;
                }
                temp->next = nNode;
        }

        return nNode;
}

// 링크드 리스트를 출력하는 함수
void printLinkedList(linkNode* head) {
        linkNode* temp = head;
        while (temp != NULL) {
                printf("        - [ %d ]번 도시에서   >   [ %d ]일 숙박\n", temp->cur_city, temp->stay);
                temp = temp->next;
        }
        printf("\n");
}

// 예약자 정보 확인 함수
void Show_reservation(int id) {
        // reservation_info[id]
        /*
        // 예약자 확인할때 필요한 구조체
        typedef struct Reservation_info {
        char my_name[100];                      // 여행자 이름
        int year;                               // 여행시작 날짜
        int month;
        int day;
        int period;                             // 여행 기간
        int budget;                             // 얘산
        int total_price;                        // 총 가격
        int total_distance;             // 총 이동거리
        int fly_ver;                    // 항공권 옵션
        int hotel_ver;                  // 호텔 옵션
        linkNode* head;                 // 도시이동 링크드 정보들
        }reservation_info[1000];
         */
        printf("\n\n\n  # 예약자 정보 #\n");
        printf("    ▶ 성함 : %s\n", reservation_info[id].my_name);
        printf("    ▶ 출발 날짜 : %d년 %d월 %d일\n", reservation_info[id].year, reservation_info[id].month, reservation_info[id].day);
        printf("    ▶ 여행 기간 : %d일간\n", reservation_info[id].period);
        printf("    ▶ 예산 : %d (KRW)\n", reservation_info[id].budget);
        printf("    ▶ 비용 : %d (KRW)\n", reservation_info[id].total_price);
        if (reservation_info[id].fly_ver == 1)                  printf("    ▶ 항공 옵션  :  ★ [Economy] 등급\n");
        else if (reservation_info[id].fly_ver == 2)             printf("    ▶ 항공 옵션  :  ★★ [Business] 등급\n");
        else                                                                                    printf("    ▶ 항공 옵션  :  ★★★ [First] 등급\n");
        if (reservation_info[id].hotel_ver == 1)                                printf("    ▶ 호텔 옵션  :  ★ [Cheapest] 호텔\n");
        else if (reservation_info[id].hotel_ver == 2)                   printf("    ▶ 호텔 옵션  :  ★★ [Resonable] 호텔\n");
        else                                                                                                    printf("    ▶ 호텔 옵션  :  ★★★ [Flex] 호텔\n");
        printf("\n\n\n ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ[AI예약 결과판]ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ\n");
        printf("\n\n  (1) 최소비용 항공 이동 경로 : ");
        linkNode* head = reservation_info[id].head;
        linkNode* temp = head;
        for (temp = head; temp != NULL; temp = temp->next)
                printf(" [%d]도시 -> ", temp->cur_city);
        printf(" [%d]도시\n\n", head->cur_city);
        printf("        - 총 이동거리 : [ %d ] km\n\n\n\n", reservation_info[id].total_distance);


        // 항공권 예약 정보 출력
        printf("  (2) [항공권 예약 정보]는 아래와 같습니다.\n\n");
        if (reservation_info[id].fly_ver == 1)                  printf("        ★ [Economy]등급으로 예약완료.\n");
        else if (reservation_info[id].fly_ver == 2)             printf("        ★★ [Business]등급으로 예약완료.\n");
        else                                                                                    printf("        ★★★ [First]등급으로 예약완료.\n");

        printf("\n            [출발지]           [도착지]                [가격]\n");
        printf("     ========================================================================\n");

        for (temp = head; temp != NULL; temp = temp->next) {
                printf("          [ %d ]번 도시   →   [ %d ]번 도시        [ %d ] KRW(원)\n", temp->cur_city, temp->next_city, temp->flight_cost);
        }
        printf("     ========================================================================\n");


        // 링크드 리스트 출력
        printf("\n  (3) 도시별 [날짜 분배]는 아래와 같습니다. \n\n");
        printLinkedList(head);


        // 호텔 예약 출력
        printf("\n\n  (4) [호텔 예약 정보]는 아래와 같이 진행했습니다.\n\n");
        if (reservation_info[id].hotel_ver == 1)                                printf("        ★ [Cheapest] 방식으로 호텔 예약 완료\n");
        else if (reservation_info[id].hotel_ver == 2)                   printf("        ★★ [Resonable] 방식으로 호텔 예약 완료\n");
        else                                                                                                    printf("        ★★★ [Flex] 방식으로 호텔 예약 완료\n");
        printf("\n            [일차]        [도시번호]               [호텔가격]\n");
        printf("     ================================================================================\n");
        int hot_cnt = 1;
        for (temp = head; temp != NULL; temp = temp->next) {            // 방문할 도시 돌면서
                for (int i = 0; i < temp->stay; i++) {                                          // 묵을 날 수 돌면서
                        printf("         - %d일차  :  [ %d ]번도시 Hotel      [ %d ] KRW(원) 에 예약완료.\n", hot_cnt, temp->cur_city, temp->hotel[i]);
                        hot_cnt++;
                }
                printf("\n\n");
        }
        printf("     ================================================================================\n");
}


int main() {
        //--------------------------------------------------------------------------------------------------------//
        //--------------------------[1단계] 도시(100,300)그래프 생성하기 -------------------------------------------//
        //--------------------------------------------------------------------------------------------------------//
        srand(time(NULL));  // 난수 발생기 초기화

        int** adjMatrix = createAdjMatrix();

        // 연결 그래프가 될 때까지 그래프 생성
        Make_Coneected_Graph(adjMatrix);
        // 직항정보 카피 해놓기
        int originMatrix[NODES][NODES];
        for (int i = 0; i < NODES; i++) {
                for (int j = 0; j < NODES; j++) {
                        originMatrix[i][j] = adjMatrix[i][j];
                }
        }


        floyd(adjMatrix);       // 플로이드 워셜 : 도시 이동의 최단거리로 모두 갱신

        // printOriginMatrix(originMatrix);  // 직항 행렬 출력
        // printAdjMatrix(adjMatrix);  // 플로이드 후(경유) 행렬 출력



        //--------------------------------------------------------------------------------------------------------//
        //--------------------------[2단계] 호텔 RB트리 + 예약자RB트리 생성하기 -------------------------------------//
        //--------------------------------------------------------------------------------------------------------//
        struct rbtree* HT[NODES];  // 100개 각 도시마다 호텔 루트를 저장하는 배열 선언
                                   // 호텔 RB tree 생성
        Make_Hotel_RBtrees(HT);

        int avail_room[NODES];          // 도시별 호텔 남은 개수
        for (int i = 0; i < NODES; i++) avail_room[i] = HOTELS; // 초기화

        // 예약자 RB트리 생성
        // RB tree 생성
        struct rbtree* BOOKT = newRBtree();
        int booked_num = 0;             // 예약 완료된 수(BOOKT의 노드개수) 초기화

        //--------------------------------------------------------------------------------------------------------//
        //--------------------------[3단계] UI 디자인 하기---------------------------------------------------------//
        //--------------------------------------------------------------------------------------------------------//
        // input(예산, 여행기간, 목적지들, 출발지, 예산버전(cheap/resonable/flex))
        // 
        // - 목적지 : 출발지 - 1 - 3 - 6 - 4 - 출발지 경로를 최소한의 거리와 비용으로 짠다.
        //                              ㄴ  랜덤 경로 1000개중 지역최적 경로 제시(극솟값)
        // 
        // - 예산 : (쌈, 적절, 비쌈) 선택에 따라서 호텔가격과 항공편(economy, buissniss, first)를 예약.
        //
        // - 여행기간 : 모든 도시별로 균등하게 분배. 남은 일자가 있을 경우 출발지부터 순서대로 하루씩 더 분배.


        // Output : customer-id, itinerary (timed information of hotels, transportations, sites, dates, total price)
        //
        // - customer-id : 예약자번호를 부여(중복x). 예약자 RB트리에 예약자 번호를 key로 INSERT.(예약 확정!)
        //
        // - sites : 방문 도시 순서 제시
        // 
        // - dates : 날짜별 방문도시 제시
        // 
        // - trasnportation : 최적 경로에 따른 항공권 이동경로, 가격을 제시
        //
        // - timed information of hotels : 여행 일자별 호텔 예약정보(도시이름, 호텔이름, 날짜별, 가격, 입실, 퇴실 시간)
        // 
        // - total price : 항공권 + 호텔비 = 총가격을 제시
        //



        // 목적지 경로 짜기, 여행기간 분배 먼저 짜보자!
        // 3-1. INPUT 받기 (budget, destination, tour-period ...)
        int cmd, cmd2;          // 명령 번호
        int id;                 // 예약자번호

        // UI 시작!!
        while (true) {
                cout << "\n\n\n\n\n\n\n\n =====================================================================================================================\n";
                cout << "  여행(AI travel) 자동 예약 시스템 [홈]입니다. 옵션을 선택해주세요.\n\n  (1.새로운 예약  /  2.예약 삭제  /  3.예약정보확인  /  " 
                        << "4.예약자번호보기  /  5.호텔정보  /  6.항공권정보  /  7.AI 챗봇 서비스  /  8.종료) \n\n      - 선택(번호)? : ";
                scanf("%d", &cmd);
                if (cmd == 8) {
                        printf("\n\n  8. 종료를 선택하셨습니다.\n");
                        break;
                }
                else if (cmd == 7) {
                        cout << "\n\n  7. [여행 추천 AI 챗봇 서비스]를 시작하겠습니다.\n  영어로 자유롭게 질문해주세요. 종료를 원하시면 [exit] 을 입력해주세요.\n\n";
                        using json = nlohmann::json;
                        openai::start();

                        string previousUserContent = "hello";
                        string previousGPTResponse = "hello! How can I help you?";

                        while (true) {
                                cout << "Q : ";
                                string userContent;
                                //cin.ignore();
                                getline(cin, userContent);
                                if (userContent == "exit") {
                                        cout << "\n  [여행 추천 AI 챗봇 서비스]를 종료합니다. 이용해주셔서 감사합니다.\n  [홈]으로 돌아갑니다.\n";
                                        break;
                                }

                                // 이전 대화와 현재 대화를 결합하여 JSON 객체 생성
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

                                // Chat API 호출
                                auto chat = openai::chat().create(jsonRequest);

                                // GPT 응답 저장
                                previousUserContent = userContent;
                                previousGPTResponse = chat["choices"][0]["message"]["content"].get<string>();

                                // 응답 출력
                                cout << "GPT : " << previousGPTResponse << '\n';
                        }
                }
                else if (cmd == 6) {
                        while (true) {
                                printf("\n\n  6. [항공권 정보]를 알려드리겠습니다.\n\n  (1.도시별 직항 가격  /  2.경유포함 최저가  /  3.나가기\n  선택(번호)? : ");
                                scanf("%d", &cmd2);
                                if (cmd2 == 1) {
                                        printf("\n\n  6-1. 도시별 직항 가격을 공지해드리겠습니다.(2차원 array 형태, - : 직항X)\n\n");
                                        printOriginMatrix(originMatrix);
                                }
                                else if (cmd2 == 2) {
                                        printf("\n\n  6-1. 경유포함 최저 가격을 공지해드리겠습니다.(2차원 array 형태)\n\n");
                                        printAdjMatrix(adjMatrix);
                                }
                                else if (cmd2 == 3) {
                                        printf("\n\n  6-3. [홈]으로 돌아갑니다.\n");
                                        break;
                                }
                                else {
                                        printf("\n\n  6-3. 선택번호 오류. 다시 선택해주세요.\n\n");
                                }
                        }

                }
                else if (cmd == 5) {
                        while (true) {
                                printf("\n\n  5. [호텔 정보]를 알려드리겠습니다.\n\n  (1.특정도시 호텔가격  /  2.모든도시 호텔가격  /  3.나가기\n  선택(번호)? : ");
                                scanf("%d", &cmd2);
                                if (cmd2 == 1) {
                                        int hot_tmp;
                                        printf("\n\n  5-1. 특정도시의 호텔가격을 공지해드리겠습니다.\n  도시번호? : ");
                                        scanf("%d", &hot_tmp);
                                        printf("\n\n\n\n  %d번 도시의 예약 가능한 호텔은 [ %d ]개 남아있습니다.\n  가격은 아래와 같습니다.\n", hot_tmp, avail_room[hot_tmp]);
                                        printf("  출력 색 구분  --->   RED : ○          BLACK : ●\n");
                                        printf("\n  %d번 도시의 호텔 가격 정보입니다. \n", hot_tmp);
                                        PRINT_BST(HT[hot_tmp]->root, NULL, false);
                                }
                                else if (cmd2 == 2) {
                                        printf("\n\n  5-2. 모든도시의 호텔가격을 공지해드리겠습니다.\n");
                                        for (int i = 0; i < NODES; i++) {
                                                printf("\n\n   %d번 도시는 [ %d ]개의 호텔이 남아있습니다.\n", i, avail_room[i]);
                                                PRINT_BST(HT[i]->root, NULL, false);
                                        }
                                        printf("  출력 색 구분  --->   RED : ○          BLACK : ●\n");
                                }
                                else if (cmd2 == 3) {
                                        printf("\n\n  5-3. [홈]으로 돌아갑니다.\n");
                                        break;
                                }
                                else {
                                        printf("\n\n  5-3. 선택번호 오류. 다시 선택해주세요.\n\n");
                                }
                        }
                }
                else if (cmd == 4) {
                        printf("\n\n  4. [예약자 현황](고유번호)을 보여드리겠습니다.\n");
                        printf("  출력 색 구분  --->   RED : ○          BLACK : ●\n");
                        // 예약자 RB트리 출력해주기
                        PRINT_BST(BOOKT->root, NULL, false);
                        printf("\n  위와 같이 총 [ %d ]명이 예약해 주셨습니다.\n\n  아무 숫자를 입력하면 [홈]으로 돌아갑니다.\n  숫자 : ", booked_num);
                        int tmp;  scanf("%d", &tmp);
                }
                else if (cmd == 3) {
                        printf("\n\n  3. [예약하신 정보를 확인]해드리겠습니다.\n\n  예약자 고유번호를 입력해주세요.\n\n  customer-id : ");
                        scanf("%d", &id);

                        // 예약자 RB트리에서 찾아보고, 존재하면 정보 출력.
                        if (rbtree_find(BOOKT, id) == NULL) {
                                printf("\n\n  [고유번호 오류] 입력하신 예약자 고유번호( %d ) 정보가 존재하지 않습니다.\n  예약자 고유번호를 다시 확인해주세요.\n\n", id);
                                printf("  아무 키를 입력하면[홈]으로 돌아갑니다.\n  아무 키 : ");
                                char tmp[10];  scanf("%s", tmp);
                                continue;
                        }

                        // id가 존재하므로 struct에서 정보 보고 출력
                        Show_reservation(id);
                }
                else if (cmd == 2) {
                        printf("\n\n  2. [예약 삭제]를 진행하겠습니다.\n\n  삭제하실 예약자 고유번호를 입력해주세요.\n\n  customer-id : ");
                        scanf("%d", &id);

                        // 예약자 RB트리에서 찾아보고, 삭제진행(예약된 호텔도 전부 다시 복구)
                        if (rbtree_find(BOOKT, id) == NULL) {
                                printf("\n\n  [고유번호 오류] 입력하신 예약자 고유번호( %d ) 정보가 존재하지 않습니다.\n  예약자 고유번호를 다시 확인해주세요.\n\n", id);
                                printf("  아무 키를 입력하면[홈]으로 돌아갑니다.\n  아무 키 : ");
                                char tmp[10];  scanf("%s", tmp);
                                continue;
                        }

                        // 예약 정보 일단 보여주기
                        Show_reservation(id);
                        printf("\n\n  ㄴ 위와 같이 예약정보가 존재합니다. 정말로 삭제하시겠습니까?(y/n)\n");
                        printf("  입력?(y/n) : ");
                        char yn[10];    scanf("%s", yn);
                        if (yn[0] == 'n' || yn[0] == 'N') {
                                printf("\n\n  삭제하지 않습니다. 아무키를 누르시면 [홈]으로 돌아갑니다.\n  아무 키 : ");
                                char tmp[10];   scanf("%s", tmp);
                                continue;
                        }


                        printf("\n\n  삭제를 진행하겠습니다.....\n");
                        // id 예약자 존재하므로 취소 진행.
                        // 호텔 취소 진행.
                        linkNode* head = reservation_info[id].head;
                        linkNode* temp = head;
                        for (temp = head; temp != NULL; temp = temp->next) {
                                for (int i = 0; i < temp->stay; i++) {
                                        RB_INSERT(HT[temp->cur_city], temp->hotel[i]);          // 호텔 rb트리로 되돌려주기
                                }
                                // 호텔 사용가능 상수도 되돌리기
                                avail_room[temp->cur_city] += temp->stay;
                        }

                        // 예약자 id 취소 진행
                        RB_DELETE(BOOKT, id);
                        // 예약자 총 수 조정
                        booked_num -= 1;

                        // 예약자 구조체도 청소
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

                        printf("  - 예약번호 ( %d )의 모든 정보가 삭제되었습니다. 감사합니다.\n", id);

                }
                else if (cmd == 1) {

                        int budget, num_dest, period, start;
                        int input_dest[100 + 5];
                        int fly_ver, hotel_ver;
                        int year, month, day;
                        char my_name[100];
                        int total_price = 0;            // 총 비용

                        while (true) {
                                printf("\n\n\n  1. 환영합니다. [새로운 예약]을 시작하겠습니다.\n");

                                printf("\n  '예산'을 입력하세요.\n      - 예산(만원) : ");
                                scanf("%d", &budget);

                                printf("\n  '목적지의 개수'를 먼저 입력하세요. (<=100)\n      - 목적지 개수 : ");
                                scanf("%d", &num_dest);
                                if (num_dest > 100 || num_dest < 1) {
                                        printf("\n  [오류] : 목적지 개수가 잘못되었습니다. (1개~100개로 입력해야함)\n");
                                        exit(0);
                                }
                                printf("\n\n  '목적지들'(도시번호)을 입력해주세요. (도시번호 0~99번)\n      - 목적지들(번호) : ");
                                for (int i = 0; i < num_dest; i++) {
                                        scanf("%d", &input_dest[i]);
                                        if (input_dest[i] < 0 || input_dest[i] > 99) {
                                                printf("\n  [오류] : 도시번호가 잘못되었습니다. 다시 입력해주세요. (0~99번으로 입력해야함)\n");
                                                exit(0);
                                        }
                                }
                                printf("\n\n  목적지들 중 '시작 도시'를 입력해주세요.\n      - 시작도시(번호) : ");
                                scanf("%d", &start);
                                if (start < 0 || start > 99) {
                                        printf("\n  [오류] : 시작도시번호가 잘못되었습니다. (0~99번으로 입력해야함)\n");
                                        exit(0);
                                }
                                // 입력한 목적지중에 시작점이 없으면 오류
                                bool start_flag = false;
                                for (int i = 0; i < num_dest; i++) {
                                        if (input_dest[i] == start)
                                                start_flag = true;
                                }
                                if (!start_flag) {
                                        printf("\n  [오류] : 시작도시가 목적지들중에 존재하지 않습니다.\n");
                                        exit(0);
                                }

                                // input_dest의 0번째 도시가 시작도시가 되도록 세팅
                                for (int i = 0; i < num_dest; i++) {
                                        if (input_dest[i] == start) {
                                                int tmp = input_dest[0];
                                                input_dest[0] = input_dest[i];
                                                input_dest[i] = tmp;
                                        }
                                }



                                printf("\n\n  여행 기간을 입력하세요.\n      - 여행 기간(일) : ");
                                scanf("%d", &period);

                                // 항공권, 호텔 가격 옵션 설정
                                while (true) {
                                        printf("\n\n  항공권 옵션을 설정하겠습니다. [항공권]은 어떻게 진행할까요? (1.Economy,  2.Business,  3.First)\n        - 입력 숫자(옵션번호) : ");
                                        scanf("%d", &fly_ver);
                                        if (fly_ver == 1 || fly_ver == 2 || fly_ver == 3)       break;
                                        else                                                                                            printf("          [항공권 옵션 오류] : 숫자를 잘못 입력하셨습니다. 1,2,3번중에서 다시 입력해주세요.\n");
                                }
                                while (true) {
                                        printf("\n\n  다음으로 호텔 옵션을 설정하겠습니다. [호텔]은 어떻게 진행할까요? (1.Cheapest,  2.Reasonable,  3.Flex)\n        - 입력 숫자(옵션번호) : ");
                                        scanf("%d", &hotel_ver);
                                        if (hotel_ver == 1 || hotel_ver == 2 || hotel_ver == 3) break;
                                        else                                                                                            printf("          [호텔 옵션 오류] : 숫자를 잘못 입력하셨습니다. 1,2,3번중에서 다시 입력해주세요.\n");
                                }

                                // 출발 날짜 입력
                                printf("\n\n  [출발 날짜]를 알려주세요.  - 년도(year) : ");
                                scanf("%d", &year);
                                printf("                            -  월(month) : ");
                                scanf("%d", &month);
                                printf("                            -  일(day) : ");
                                scanf("%d", &day);

                                // 여행자 이름 받기
                                printf("\n\n  마지막으로 [여행자 이름]을 입력해주세요.\n        - 여행자 이름(영문으로) : ");
                                scanf("%s", my_name);


                                printf("\n\n\n  소중한 정보들을 입력해주셔서 감사합니다. 입력하신 정보들로 자동 예약을 진행할까요? (y/n)\n  입력(y/n) : ");
                                char yn[10];
                                scanf("%s", yn);
                                if (yn[0] == 'y' || yn[0] == 'Y')
                                        break;
                                else
                                        printf("  다시 예약하기를 선택하셨습니다. 예약 처음으로 돌아갑니다.\n\n");
                        }       // input받기 끝

                        // 3-2. 목적지 경로(방문순서)짜기
                        int* sorted_dest = make_dest_order(num_dest, input_dest, start, adjMatrix);
                        int total_distance = Sum_distance_dest_order(sorted_dest, num_dest, adjMatrix);

                        // 경로 출력해보기
                        printf("\n\n\n ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ[AI예약 결과판]ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ\n");
                        printf("\n\n  (1) 최소비용 항공 이동 경로 : ");
                        for (int i = 0; i < num_dest; i++)
                                printf(" [%d]도시 -> ", sorted_dest[i]);
                        printf(" [%d]도시\n\n", start);
                        printf("        - 총 이동거리 : [ %d ] km\n\n\n\n", total_distance);


                        // 3-3. 이동 도시별 링크드 리스트 만들기
                        linkNode* ct_head = NULL;
                        linkNode* temp = ct_head;

                        // 노드 추가 & 항공권 예약 정보 출력
                        printf("  (2) [항공권 예약 정보]는 아래와 같습니다.\n\n");
                        if (fly_ver == 1)                       printf("        ★ [Economy]등급으로 예약완료.\n");
                        else if (fly_ver == 2)          printf("        ★★ [Business]등급으로 예약완료.\n");
                        else                                            printf("        ★★★ [First]등급으로 예약완료.\n");

                        printf("\n            [출발지]           [도착지]                [가격]\n");
                        printf("     ========================================================================\n");

                        for (int i = 0; i < num_dest; i++) {
                                // 현재도시, 다음도시, 머무는날수, 항공편, 항공가격, 호텔예약을 저장
                                temp = addNode(&ct_head, sorted_dest, period / num_dest, i, num_dest, fly_ver, adjMatrix);
                                printf("          [ %d ]번 도시   →   [ %d ]번 도시        [ %d ] KRW(원)\n", temp->cur_city, temp->next_city, temp->flight_cost);
                                total_price += temp->flight_cost;                               // 총 가격 누적
                        }
                        printf("     ========================================================================\n");
                        // 여행기간 분배하기(남은일자 있으면 앞에서부터 하루씩 증가)
                        int rest_day = period % num_dest;
                        if (rest_day > 0) {
                                temp = ct_head;
                                while (temp != NULL) {
                                        temp->stay += 1;
                                        rest_day--;
                                        if (rest_day == 0)
                                                break;          // 다 나눠줬으면 종료
                                        temp = temp->next;
                                }
                        }



                        // 링크드 리스트 출력
                        printf("\n  (3) 도시별 [날짜 분배]는 아래와 같이 진행합니다. \n\n");
                        printLinkedList(ct_head);


                        // 3-4. 호텔 예약하기 (도시별 여행기간을 보고)
                        // 예약자가 900명이상 : 에러
                        if (booked_num >= 900) {
                                printf("\n  [예약자 폭주] 죄송합니다. 예약 완료자가 900명이 넘어서 더이상 예약이 불가능합니다.\n  예약은 파기됩니다....\n\n");
                                printf("  아무 숫자나 누르면 [홈]으로 돌아갑니다.\n  아무 숫자 : ");
                                int tmp;  scanf("%d", &tmp);
                                continue;
                        }
                        // 갈 도시들 둘러보면서
                        bool room_flag = true;
                        for (temp = ct_head; temp != NULL; temp = temp->next) {
                                // 그 도시에 호텔수가 부족하다면(하루에 호텔 하나를 예약함)
                                if (temp->stay > avail_room[temp->cur_city]) {
                                        room_flag = false;
                                        break;
                                }
                        }
                        // 호텔수가 부족하므로 예약 파기
                        if (!room_flag) {
                                printf("\n  [호텔 부족] 죄송합니다. %d번 도시에 예약가능한 호텔이 충분치 않습니다.\n  다른 도시를 선택하거나 여행일자를 줄여주세요.\n  예약은 파기됩니다....\n\n", temp->cur_city);
                                printf("  아무 숫자나 누르면 [홈]으로 돌아갑니다.\n  아무 숫자 : ");
                                int tmp;  scanf("%d", &tmp);
                                continue;
                        }

                        // 호텔 예약 진행
                        printf("\n\n  (4) [호텔 예약 정보]는 아래와 같이 진행했습니다.\n\n");
                        if (hotel_ver == 1)                             printf("        ★ [Cheapest] 방식으로 호텔 예약 완료\n");
                        else if (hotel_ver == 2)                printf("        ★★ [Resonable] 방식으로 호텔 예약 완료\n");
                        else                                                    printf("        ★★★ [Flex] 방식으로 호텔 예약 완료\n");
                        printf("\n            [일차]        [도시번호]               [호텔가격]\n");
                        printf("     ================================================================================\n");
                        int hot_cnt = 1;
                        for (temp = ct_head; temp != NULL; temp = temp->next) {         // 방문할 도시 돌면서
                                for (int i = 0; i < temp->stay; i++) {                                          // 묵을 날 수 돌면서
                                                                                                                // cheapest, resonable, flex옵션 고려
                                        struct node_t* booking_hotel;
                                        if (hotel_ver == 1)                             booking_hotel = MinKeyOfRBtree(HT[temp->cur_city]);             // 1) cheapest (left most)
                                        else if (hotel_ver == 2)                booking_hotel = HT[temp->cur_city]->root;                               // 2) resonable (root)
                                        else                                                    booking_hotel = MaxKeyOfRBtree(HT[temp->cur_city]);             // 3) flex (right most)


                                        if (booking_hotel) {
                                                // 호텔나무에서 삭제(예약확정)
                                                int booking_price = booking_hotel->key;
                                                // printf("  도시 : %d,  지우려는 key가격 : %d\n", temp->city_num, cheap_hotel->key);
                                                RB_DELETE(HT[temp->cur_city], booking_hotel->key);
                                                avail_room[temp->cur_city] -= 1;        // 예약가능한 호텔 수 1개 감소
                                                temp->hotel[i] = booking_price;         // 묵을 호텔 가격(key) 기록
                                                printf("         - %d일차  :  [ %d ]번도시 Hotel      [ %d ] KRW(원) 에 예약완료.\n", hot_cnt, temp->cur_city, booking_price);
                                                hot_cnt++;
                                                total_price += booking_price;                   // 총 비용 누적
                                        }
                                }
                                printf("\n\n");
                        }
                        printf("     ================================================================================\n");



                        // Output : customer-id, itinerary (timed information of hotels, transportations, sites, dates, total price)
                        //
                        // - customer-id : 예약자번호를 부여(중복x). 예약자 RB트리에 예약자 번호를 key로 INSERT.(예약 확정!)
                        //
                        // - sites : 방문 도시 순서 제시
                        // 
                        // - dates : 날짜별 방문도시 제시
                        // 
                        // - trasnportation : 최적 경로에 따른 항공권 이동경로, 가격을 제시
                        //
                        // - timed information of hotels : 여행 일자별 호텔 예약정보(도시이름, 호텔이름, 날짜별, 가격, 입실, 퇴실 시간)
                        // 
                        // - total price : 항공권 + 호텔비 = 총가격을 제시
                        // 




                        // struct에 모든 종합정보를 다 담으면 될 것 같다.
                        // 위에 프린트 정보 다시 그대로 뽑을 수 있도록. (list_head , total_distance , hotel_ver, fly_ver, customer-id, name, start_date)로 struct의 요소를 구성.
                        //                                                                                                      ㄴ list_head에 도시이동경로, 항공권가격, 호텔가격, 머문날수가 기록되어 있다.

                        // rb트리 노드에 추가하는건 무리. 그냥 struct[100][100]짜리 만들자.
                        // 예산 체크하고 예산에 맞으면 예약진행해야함.
                        // 일단 1.예산체크 2.예약확인 3.예약삭제 구현하고 보고서쓰자.

                        printf("\n\n    ▷ 총 예산 : %d (KRW)\n", budget);
                        printf("    ▶ 총 비용 : %d (KRW)\n\n", total_price);
                        // 예산이 모자라면 호텔 예약 삭제 진행하고 예약파기
                        if (budget < total_price) {
                                // 호텔 취소 진행.
                                for (temp = ct_head; temp != NULL; temp = temp->next) {
                                        for (int i = 0; i < temp->stay; i++) {
                                                RB_INSERT(HT[temp->cur_city], temp->hotel[i]);          // 호텔 rb트리로 되돌려주기
                                        }
                                        // 호텔 사용가능 상수도 되돌리기
                                        avail_room[temp->cur_city] += temp->stay;
                                }
                                printf("\n  [예산 부족] 죄송합니다. 예산보다 총 예상비용이 더 나오므로 예약이 불가능합니다.\n  예산 : %d (KRW)\n  예상 총비용 : %d (KRW)\n  예산을 늘려주시거나 호텔, 비행 옵션을 낮춰서 다시 예약해주세요.\n  예약은 파기됩니다....\n\n", budget, total_price);
                                printf("  아무 숫자나 누르면 [홈]으로 돌아갑니다.\n  아무 숫자 : ");
                                int tmp;  scanf("%d", &tmp);
                                continue;
                        }

                        // 3-5. customer-id 생성 & 예약자RB트리에 삽입
                        int random_customer_id = 0;
                        do {
                                random_customer_id = (rand() % 900) + 100;              // 100 ~ 999번의 예약번호
                        } while (rbtree_find(BOOKT, random_customer_id) != NULL);

                        // customer-id를 예약자RB tree에 삽입
                        RB_INSERT(BOOKT, random_customer_id);
                        booked_num += 1;                // 예약자 1명 증가

                        // 3-6. 예약정보 종합해서 struct에 기록
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



                        printf("\n ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ\n");
                        printf("\n\n  위와 같이 항공권과 호텔 예약이 완료되었습니다.\n\n  고객님의 예약번호는  (customer-id) : [ %d ] 입니다.\n\n  감사합니다. 아무 숫자를 입력하면 [홈]으로 돌아갑니다.\n  아무 숫자 : ", random_customer_id);
                        int tmp;  scanf("%d", &tmp);

                }// 1.새로운 예약 끝
                else {
                        printf("\n\n  [명령어 오류] : 명령어를 잘못 입력하셨습니다. 다시 입력해주세요.\n\n");
                        printf("  아무 숫자나 누르면 [홈]으로 돌아갑니다.\n  아무 숫자 : ");
                        int tmp;  scanf("%d", &tmp);
                }
        }// 명령어 while문 끝

        //--------------------------------------------------------------------------------------------------------//
        //--------------------------[마지막 단계] 메모리 해제하기 --------------------------------------------------//
        //--------------------------------------------------------------------------------------------------------//
        // 생성된 호텔루트들을 메모리에서 해제
        printf("\n  사용했던 '메모리를 해제중'입니다. 잠시만 기다려주세요....\n\n");
        for (int i = 0; i < 100; i++) {
                free(HT[i]);
        }
        printf("\n  '종료'하겠습니다. 다시 이용해주세요. 감사합니다.\n");
        freeAdjMatrix(adjMatrix);   // 인접행렬 메모리 해제

        return 0;
}



