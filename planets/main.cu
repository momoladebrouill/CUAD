#include "raylib.h"
#include <stdio.h>
#define HEIGHT 800
#define WIDTH 800
#define N 255
#define DT 0.1
#define K 1e3
#define ALPHA 0.1
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

__global__ void update(point *points,float midx, float midy) {
    int i = threadIdx.x;
    // Compute acceleration
    add(&points[i].pos,&points[i].acc);
    // Update position
    point p = points[i];
    vec * old_p = (vec*) malloc(sizeof(vec));
    copy(old_p,&p.pos);
    vec v = {p.pos.x - p.oldpos.x, p.pos.y - p.oldpos.y};
    float uy = p.pos.y - midy;
    float ux = p.pos.x - midx;
    float dist_squared = uy*uy + ux*ux;
    uy = uy/sqrt(dist_squared);
    ux = ux/sqrt(dist_squared);
    p.acc.y = -K*uy/(dist_squared) - ALPHA*v.y;
    p.acc.x = -K*ux/(dist_squared) - ALPHA*v.x;
    mult(&p.pos, 2.0);
    mult(&p.oldpos,-1.0);
    add(&p.pos,&p.oldpos);
    mult(&p.acc,DT*DT);
    add(&p.pos, &p.acc);
    copy(&p.oldpos, old_p);

    if(p.pos.y > HEIGHT){
      p.pos.y-= HEIGHT;
      p.oldpos.y-= HEIGHT;
    }
    if(p.pos.y < 0){
      p.pos.y+= HEIGHT;
      p.oldpos.y+= HEIGHT;
    }
    if(p.pos.x > WIDTH){
      p.pos.x-= WIDTH;
      p.oldpos.x-= WIDTH;
    }
    if(p.pos.x < 0){
      p.pos.x+= WIDTH;
      p.oldpos.x+= WIDTH;
    }
    points[i] = p;
    free(old_p);
 
}

__managed__ point d_points[N+1];


__host__ int main(){
  SetTraceLogLevel(5); 

  for (int i = 0; i < N; i++) {
    float x = i*WIDTH/N;
    float y = abs(rand()) % HEIGHT;
    d_points[i].pos.x = x;
    d_points[i].pos.y = y;
    d_points[i].oldpos.x = x + (float)rand()/(float)(RAND_MAX/5)-2.5;
    d_points[i].oldpos.y = y + (float)rand()/(float)(RAND_MAX/5)-2.5;
    d_points[i].acc.y = 0;
    d_points[i].acc.x = 0;
  }


  InitWindow(WIDTH, HEIGHT, "i don't know");
  SetTargetFPS(60);               // Set our game to run at 60 frames-per-second
  cudaError_t cudaError;
  while (!WindowShouldClose() && ((cudaError = cudaGetLastError()) == cudaSuccess)){
      BeginDrawing();
      DrawRectangle(0, 0, WIDTH, HEIGHT, Fade(BLACK, 0.1));
      float midx=0,midy=0;
      for(int i=0;i<N-1;i++){
        DrawCircle(d_points[i].pos.x, d_points[i].pos.y, 5.0, RED);
        midx+=d_points[i].pos.x;
        midy+=d_points[i].pos.y;
      }
      char s[] = "FPS: xxx";
      sprintf(s,"FPS: %i", GetFPS());
      DrawText(s, 10, 10, 20, RED);
      DrawCircle(midx/N, midy/N,10.0, BLUE);
      EndDrawing();

      update<<<1, N>>>(d_points,midx/N,midy/N);
      cudaDeviceSynchronize();
  }
  CloseWindow();
  if(cudaError != cudaSuccess)
    printf("\e[31m%s\e[0m \n", cudaGetErrorName(cudaError));
  return 0;
}
