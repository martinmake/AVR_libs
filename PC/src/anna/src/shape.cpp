#include "shape.h"

namespace Anna
{
	const Shape Shape::INVALID = Shape(0, 0, 0, 0);

	Shape::Shape(void)
		: Shape(INVALID)
	{
	}
	Shape::Shape(uint64_t initial_width, uint64_t initial_height, uint64_t initial_depth, uint64_t initial_time)
		: m_width(initial_width), m_height(initial_height), m_depth(initial_depth), m_time(initial_time)
	{
	}

	Shape::~Shape(void)
	{
	}
}
