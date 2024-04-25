#include "raylib.h"
#include <stdio.h>
#define HEIGHT 800
#define WIDTH 800
#define W 32
#define H 32
#define SIZE WIDTH/W
using namespace std;



/* __device__ pour dire c'est sur le gpu */

__global__ void update(double *valeurs, double shiftx, double shifty , double zoom) {
    int i = threadIdx.x;
    double x = i/W;
    double y = i % H;
    double cre = (double)x/(double)W * zoom + shiftx;
    double cim = (double)y/(double)H * zoom+ shifty;
    double zre = 0;
    double zim = 0;
    int n = 0;
    int max_iter = 100/zoom;
    while (zre*zre + zim*zim < 4.0 && n < max_iter) {
        /* z = z*z + c; */
        double zre_new = zre*zre - zim*zim + cre;
        double zim_new = 2*zre*zim + cim;
        zre = zre_new;
        zim = zim_new;
        n++;
    }
    valeurs[i] = n >= max_iter ? -1 : (double)n/max_iter;
}

__managed__ double valeurs[W*H];


__host__ int main(){
  SetTraceLogLevel(5); 
  printf("Hello world\n");
  for (int i = 0; i < W*H; i++) {
    valeurs[i] = 0.5;
  }

  InitWindow(WIDTH, HEIGHT, "mandelprout hihi");
  SetTargetFPS(60);               // Set our game to run at 60 frames-per-second
  cudaError_t cudaError;
  double shiftx = 0.0;
  double shifty = 0.0;
  double zoom = 1.0;
  while (!WindowShouldClose() && ((cudaError = cudaGetLastError()) == cudaSuccess)){
      BeginDrawing();
      ClearBackground(BLACK);
      for(int x=0;x<W;x++){
        for(int y=0;y<H;y++){
          int i = x*W + y;
          Color c = (valeurs[i] == -1) ? BLACK : ColorFromHSV(360*valeurs[i],1,1);
          DrawRectangle(x*SIZE, y*SIZE, SIZE,SIZE, c);
        }
      }
      if (IsKeyDown(KEY_LEFT))
          shiftx -= 0.01 * zoom;
      if (IsKeyDown(KEY_RIGHT))
          shiftx += 0.01 * zoom;
      if (IsKeyDown(KEY_UP))
          shifty -= 0.01 * zoom;
      if (IsKeyDown(KEY_DOWN))
          shifty += 0.01 * zoom;
      if (IsKeyDown(KEY_KP_ADD))
          zoom *= 1.01;
      if (IsKeyDown(KEY_KP_SUBTRACT))
          zoom /= 1.01;
      char s[] = "FPS: xxx";
      sprintf(s,"FPS: %i", GetFPS());
      DrawText(s, 10, 10, 20, WHITE);
      EndDrawing();

      update<<<1,W*H>>>(valeurs, shiftx, shifty, zoom);
      cudaDeviceSynchronize();
  }
  CloseWindow();
  if(cudaError != cudaSuccess)
    printf("\e[31m%s\e[0m \n", cudaGetErrorName(cudaError));
  return 0;
}
