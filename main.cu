#include "raylib.h"
#include <stdio.h>

__global__ void add(int *a, int *b, int *c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] = a[i] + b[i];
}

__managed__ int vector_a[256], vector_b[256], vector_c[256];

int main(){
  // Make CUDA work
  for (int i = 0; i < 256; i++) {
    vector_a[i] = i;
    vector_b[i] = 256-i;
  }
  /* blocks and threads per block */
  add<<<1, 256>>>(vector_a, vector_b, vector_c);
  cudaDeviceSynchronize();

  int result_sum = 0;
  for (int i = 0; i < 256; i++) {
    result_sum += vector_c[i];
  }

  printf("Sum of vector_a and vector_b is %d\n", result_sum);
  printf("Expected result is 256*255/2 = 32640\n");
  // Make raylib work
  const int screenWidth = 800;
  const int screenHeight = 450;

  InitWindow(screenWidth, screenHeight, "raylib [core] example - basic window");
  SetTargetFPS(60);               // Set our game to run at 60 frames-per-second
  while (!WindowShouldClose())    // Detect window close button or ESC key
  {
      BeginDrawing();
      ClearBackground(RAYWHITE);
      DrawText("Congrats! You created your first window!", 190, 200, 20, LIGHTGRAY);
      EndDrawing();
  }
  CloseWindow();        
  return 0;
}
