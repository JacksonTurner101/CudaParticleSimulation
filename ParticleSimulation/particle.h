#pragma once


#ifndef PARTICLE_H
#define PARTICLE_H

#include <cuda_runtime.h>

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

#endif 
