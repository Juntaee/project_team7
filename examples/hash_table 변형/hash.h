#include <iostream>

using namespace std;

template <class T>
class node{
    public:
        T data;
        node<T>* next_node;
        node(T data_in, node<T>* next):
            data(data_in), next_node(next){}
        ~node() {cout << "remove node!"<< endl;}
};

template <class T>
class Linked_list{
    public:
        Linked_list();
        ~Linked_list();

        void addNode(T data_in);

        int get_size(){return size;}
        node<T>* get_head(){return head;}
        node<T>* get_tail(){return tail;}

    private:
        node<T>* head;
        node<T>* tail;
        int size;
 };

class HashTable{
    public:
        HashTable(int m, int t);
        ~HashTable();
        void insertItems(int k, int s, char g, double p);
        void print_hash_map();

    private:
        int table_size;
        int max_size;
        Linked_list<int>* student_id;
        Linked_list<char>* grade;
        Linked_list<double>* project_score;
};


