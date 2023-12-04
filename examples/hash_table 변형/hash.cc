#include "hash.h"

using namespace std;

template <class T>
Linked_list<T>::Linked_list() : head(nullptr), tail(nullptr), size(0) {}

template <class T>
Linked_list<T>::~Linked_list() {
	if (size > 0) {
		for (int i = 0; i < size; i++) {
			node<T>* tmp = head;
			head = head->next_node;
			delete tmp;
		}
	}
}

template <class T>
void Linked_list<T>::addNode(T data_in) {
	node<T>* new_node = new node<T>(data_in, nullptr);

	if (head == nullptr) {
		head = new_node;
		tail = new_node;
	}
	else {
		tail->next_node = new_node;
		tail = new_node;
	}
	size++;
}

HashTable::HashTable(int m, int t) : table_size(t), max_size(m) {
	student_id = new Linked_list<int>[table_size]();
	grade = new Linked_list<char>[table_size]();
	project_score = new Linked_list<double>[table_size]();
}

HashTable::~HashTable() {
	delete[] student_id;
	delete[] grade;
	delete[] project_score;
}

void HashTable::insertItems(int k, int s, char g, double p) {
	int bucket_index = k % table_size;
	int cur_size = student_id[bucket_index].get_size();

	if (cur_size < max_size) {
		student_id[bucket_index].addNode(s);
		grade[bucket_index].addNode(g);
		project_score[bucket_index].addNode(p);
	}
	else {
		cout << "Delete data that exceeds max size." << endl;
		cout << "Data: " << s << ", " << g << ", " << p << endl;
	}
}

void HashTable::print_hash_map() {
	for (int i = 0; i < table_size; i++) {
		cout << i << " -> ";
		int cur_size = student_id[i].get_size();
		node<int>* id_tmp = student_id[i].get_head();
		node<char>* grade_tmp = grade[i].get_head();
		node<double>* score_tmp = project_score[i].get_head();
		for (int j = 0; j < cur_size; j++) {
			cout << id_tmp->data << ", " << grade_tmp->data << ", " << score_tmp->data
				<< " -> ";
			id_tmp = id_tmp->next_node;
			grade_tmp = grade_tmp->next_node;
			score_tmp = score_tmp->next_node;
		}
		cout << endl;
	}
}