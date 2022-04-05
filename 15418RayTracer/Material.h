#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H
#include "Defined.h"
#include <stdlib.h>


class Material
{
public:
	Material(Color3f a, Color3f e) {
		albedo = a;
		emitted = e;
	}
	Material() {
	}
	Color3f albedo;
	Color3f emitted;


};

#endif