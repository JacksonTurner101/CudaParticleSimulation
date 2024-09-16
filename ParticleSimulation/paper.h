#pragma once


#ifndef PAPER_H
#define PAPER_H

#include <cuda_runtime.h>

struct Paper {
    float3 position;
    float3* colourValues;
    float3 scale;

};
#endif 
