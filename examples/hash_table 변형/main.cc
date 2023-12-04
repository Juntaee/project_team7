#include "hash.h"

int main(int argc, char *argv[])
{
    int push_num = 13;
    int key[13] = {467178136, 40614793, 580233934, 943823896, 860077417, 202733146, 227550872, 639275989, 214011427, 85963767, 8, 25, 42};
    int student_id[13] = {36120785, 71846324, 94669195, 94274211, 10812767,  3915265, 87024034, 6946644, 86037810, 37909621, 2542325, 467425, 453664};
    char grade[13] = {'A', 'B', 'A', 'C', 'F', 'A', 'B', 'A', 'A', 'B', 'C', 'B', 'A'};
    double project_score[13] = {92.1, 90.3, 97.4, 86.4, 0.0, 95.3, 87.4, 99.3, 98.1, 85.4, 78.5, 87.5, 99.8};
    
    int max_size = 4;
    int table_size = 17;
    HashTable h(max_size, table_size);
    
    for(int i=0;i<push_num;i++)
        h.insertItems(key[i], student_id[i], grade[i], project_score[i]);
    h.print_hash_map();

    return 0;
}



