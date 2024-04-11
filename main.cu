#include "raylib.h"
#include <stdio.h>
#define HEIGHT 450
#define WIDTH 800
#define N 254
#define DT 0.1
using namespace std;

typedef struct vec {
      float x;
      float y;
} vec;

typedef struct point {
     vec pos;
     vec oldpos;
     vec acc;
} point;

__device__ void add(vec * a, vec * b) {
    a->x += b->x;
    a->y += b->y;
}

__device__ void mult(vec * a, float b) {
    a->x *= b;
    a->y *= b;
}

__device__ void copy(vec * dst, vec * src) {
    dst->x = src->x;
    dst->y = src->y;
}

__global__ void update(point *points) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // Compute acceleration
    add(&points[i].pos,&points[i].acc);
    // Update position
    point p = points[i];
    vec * old_p = (vec*) malloc(sizeof(vec));
    copy(old_p,&p.pos);

    p.acc.y = 1.0;
    mult(&p.pos, 2.0);
    mult(&p.oldpos,-1.0);
    add(&p.pos,&p.oldpos);
    mult(&p.acc,DT*DT);
    add(&p.pos, &p.acc);
    copy(&p.oldpos, old_p);

    if(p.pos.y > HEIGHT){
      p.pos.y = 0;
      p.oldpos.y = 0;
    }
    points[i] = p;
 
}

__managed__ point d_points[N];


__host__ int main(){
  SetTraceLogLevel(5); 

  for (int i = 0; i < N; i++) {
    float y = abs(rand()) % HEIGHT;
    d_points[i].pos.x = i * 3.0;
    d_points[i].pos.y = y;
    d_points[i].oldpos.x = i * 3.0;
    d_points[i].oldpos.y = y;
    d_points[i].acc.y = 0;
    d_points[i].acc.x = 0;
  }


  InitWindow(WIDTH, HEIGHT, "i don't know");
  SetTargetFPS(60);               // Set our game to run at 60 frames-per-second
  while (!WindowShouldClose())    // Detect window close button or ESC key
  {
      BeginDrawing();
      ClearBackground(BLACK);
      for(int i=0;i<N;i++)
        DrawCircle(d_points[i].pos.x, d_points[i].pos.y, 5.0, BLUE);
      EndDrawing();
      update<<<1, N>>>(d_points);
      printf("debut");
      cudaDeviceSynchronize();
      printf(" fin");
  }
  CloseWindow();        
  return 0;
}
