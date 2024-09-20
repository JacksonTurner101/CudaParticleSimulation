#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Image.h"
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <random>
#include <cmath> 

struct Particle {
    float3 position;
    float3 velocity;
    float3 colour;
    bool hasCollided;

    __device__ void update(float dt) {
        position.x += velocity.x * dt;
        position.y += velocity.y * dt;
        position.z += velocity.z * dt;
    }
};

struct Paper {
    float3 position;
    float3 scale;
    float3* colourValues;
};

__device__ float atomicAddFloat(float* address, float value) {
    float old = *address, assumed;
    do {
        assumed = old;
        old = atomicCAS((int*)address, __float_as_int(assumed), __float_as_int(value + assumed));
    } while (assumed != old);
    return old;
}

__global__ void moveParticleKernel(Particle* particles, float dt, float3* paperColourValues, float3 paperPosition, float3 paperScale, int paperWidth, int paperHeight) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 1000) return;  
    if (!particles[index].hasCollided) {
        particles[index].update(dt);
        // Check for collision
        if (particles[index].position.x > paperPosition.x - paperScale.x / 2 && particles[index].position.x < paperPosition.x + paperScale.x / 2 &&
            particles[index].position.y > paperPosition.y - paperScale.y / 2 && particles[index].position.y < paperPosition.y + paperScale.y / 2 &&
            particles[index].position.z > paperPosition.z - paperScale.z / 2 && particles[index].position.z < paperPosition.z + paperScale.z / 2) {
            // Collision has happened
            particles[index].hasCollided = true;

            //calculate where the start of the paper is
            float paperStartColourValueArrayX = paperPosition.x - (paperScale.x / 2);
            float paperStartColourValueArrayY = paperPosition.z - (paperScale.z / 2);

            float xConstant = paperWidth / paperScale.x;
            float yConstant = paperHeight / paperScale.z;

            int indexX = ((particles[index].position.x - paperStartColourValueArrayX) * xConstant);
            int indexY = ((particles[index].position.z - paperStartColourValueArrayY) * yConstant);

            if (indexX >= 0 && indexX < paperWidth && indexY >= 0 && indexY < paperHeight) {
                int idx = indexY * paperWidth + indexX;
                float b = 0.1f; //blending factor

                //These are different methods of adding the color value of the particle to the paper

                /*atomicAddFloat(&paperColourValues[idx].x, b * (particles[index].colour.x - paperColourValues[idx].x));
                atomicAddFloat(&paperColourValues[idx].y, b * (particles[index].colour.y - paperColourValues[idx].y));
                atomicAddFloat(&paperColourValues[idx].z, b * (particles[index].colour.z - paperColourValues[idx].z));*/

                //paperColourValues[idx].x = ((1.0f - b) * paperColourValues[idx].x + b * particles[index].colour.x);
                //paperColourValues[idx].y = ((1.0f - b) * paperColourValues[idx].y + b * particles[index].colour.y);
                //paperColourValues[idx].z = ((1.0f - b) * paperColourValues[idx].z + b * particles[index].colour.z);

                atomicAdd(&paperColourValues[idx].x, 0.5f);
                
                //paperColourValues[idx].x = 1.0f;
                //paperColourValues[idx].y = 0;
                //paperColourValues[idx].z = 0;
            }
        }
    }
}

//how many particles in the spray can
const int particleArraySize = 20000;

//paper dimensions
const int paperWidth = 1000;
const int paperHeight = 1000;

//time in seconds
const float totalSimulationTime = 2.0f;

int main()
{

    //initialising paper values
    Paper paper;
    paper.position = { 0.0f, -0.2f, 0.0f };
    paper.scale = { 1.0f, 0.5f, 1.0f };
    paper.colourValues = new float3[paperWidth * paperHeight];
    for (int i = 0; i < paperWidth * paperHeight; i++) {
        //setting all paper colour values to white
        paper.colourValues[i] = { 0.0f, 0.0f, 0.0f };

    }

    std::mt19937 rd;
    std::uniform_real_distribution<float> radiusDist(0.0, 0.3);   // random radius
    std::uniform_real_distribution<float> angleDist(0.0, 2 * 3.14); // random angle


    //initailize array of particles (filling the spray can)
    Particle particles[particleArraySize];
    for (int i = 0; i < particleArraySize; i++) {

        float radius = radiusDist(rd);
        float angle = angleDist(rd);

        float xVal = radius * cos(angle);
        float zVal = radius * sin(angle);

        particles[i].position = { xVal , 0.0f, zVal};
        particles[i].velocity = { 0.0f, -1.0f, 0.0f };
        particles[i].colour = { 1.0f, 0.0f, 0.0f }; // red color
        particles[i].hasCollided = false;
    }

    //pointers
    Particle* dev_particles = nullptr;
    float3* dev_paperColourValues = nullptr;

    //allocate memory
    cudaMalloc((void**)&dev_particles, particleArraySize * sizeof(Particle));
    cudaMalloc((void**)&dev_paperColourValues, paperWidth * paperHeight * sizeof(float3));

    //copy data from cpu to gpu
    cudaMemcpy(dev_particles, particles, particleArraySize * sizeof(Particle), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_paperColourValues, paper.colourValues, paperWidth * paperHeight * sizeof(float3), cudaMemcpyHostToDevice);

    //calculating delta time
    auto simStart = std::chrono::high_resolution_clock::now();
    auto start = simStart;
    auto end = simStart;
    std::chrono::duration<float> elapsed;
    float totalElapsedTime = 0.0f;

    //int whileLoopsNumber = 0;

    //run simulation for 2 seconds
    while (totalElapsedTime < totalSimulationTime) {
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        float dt = elapsed.count();
        start = end;

        totalElapsedTime += dt;

        moveParticleKernel << <(particleArraySize + 255) / 256, 256 >> > (dev_particles, dt, dev_paperColourValues, paper.position, paper.scale, paperWidth, paperHeight);
        cudaDeviceSynchronize();
        //whileLoopsNumber++;
        //std::cout << "while loops num = " << whileLoopsNumber << std::endl;
    }

    cudaMemcpy(particles, dev_particles, particleArraySize * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaMemcpy(paper.colourValues, dev_paperColourValues, paperWidth * paperHeight * sizeof(float3), cudaMemcpyDeviceToHost);

    cudaFree(dev_particles);
    cudaFree(dev_paperColourValues);

    //// Print all values in paper.colourValues
    //for (int y = 0; y < paperHeight; ++y) {
    //    for (int x = 0; x < paperWidth; ++x) {
    //        float3 colourValue = paper.colourValues[y * paperWidth + x];
    //        std::cout << "paper.colourValues[" << y << "][" << x << "] = ("
    //            << colourValue.x << ", "
    //            << colourValue.y << ", "
    //            << colourValue.z << ")\n";
    //    }
    //}

    int centerIndexX = paperWidth / 2;
    int centerIndexY = paperHeight / 2;
    int centerIndex = centerIndexY * paperWidth + centerIndexX;

    float3 centerColor = paper.colourValues[centerIndex];
    std::cout << "Color value at the center of the paper: ("
        << centerColor.x << ", "
        << centerColor.y << ", "
        << centerColor.z << ")" << std::endl;


    //const int width = 640;
    //const int height = 480;

    Image image(paperWidth, paperHeight);

    for (int y = 0; y < paperHeight; y++) {
        for (int x = 0; x < paperWidth; x++) {
            
            Color color(paper.colourValues[x + y * paperWidth].x, paper.colourValues[x + y * paperWidth].y, paper.colourValues[x + y * paperWidth].z);
            //image.SetColor(Color((float)x / (float)width, 1.0f - ((float)x / (float)width), (float)y / (float)height), x, y);
            image.SetColor(color,x,y);
        }
    }

    image.Export("image.bmp");

    delete[] paper.colourValues;

    return 0;
}
