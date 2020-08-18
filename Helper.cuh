#ifndef DESKTOP_HELPER_CUH
#define DESKTOP_HELPER_CUH

struct dynamic_array_struct {
    int k;
    int *i;
    int n;
};

struct static_array_struct {
    int k;
    int i[10];
    int n;
};

struct static_array2D_struct {
    int k;
    int i[10][10];
    int n;
};

struct inner {
    int x;
    int y;
};

struct struct_struct {
    int a;
    struct inner i;
};

#endif //DESKTOP_HELPER_CUH
