#include <gra/shader.h>

#include "gpl/static_initializer.h"
#include "gpl/primitives/all.h"
#include "gpl/primitive.h"

#include "primitives_shader.h"

namespace Gpl
{
	StaticInitializer::StaticInitializer(void)
	{
		static bool is_initialized = false;
		if (!is_initialized)
		{
			initialize();
			is_initialized = true;
		}
	}
	StaticInitializer::~StaticInitializer(void)
	{
	}

	void StaticInitializer::initialize(void)
	{
		Primitive::s_shader = *new Gra::Shader(VERTEX_SHADER, FRAGMENT_SHADER);
		Primitive::Point::s_vertex_buffer_layout.push<float>(3);
		unsigned int point_index_buffer = 0;
		Primitive::Point::s_index_buffer = *new Gra::IndexBuffer(&point_index_buffer, 1);
		glCall(glEnable(GL_PROGRAM_POINT_SIZE));
	}
}
