#pragma once
#include "Defined.h"
#include <stdlib.h>


class Material
{
public:
	Material(Color3 a, Color3 e) {
		albedo = a;
		emitted = e;
	}
	Material() {
	}
	Color3 albedo;
	Color3 emitted;


};
