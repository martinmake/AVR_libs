#include "shape.h"

namespace Anna
{
	const Shape Shape::INVALID = Shape(0, 0, 0, 0);

	Shape::Shape(void)
		: Shape(INVALID)
	{
	}
	Shape::Shape(uint16_t initial_width, uint16_t initial_height, uint16_t initial_channel_count, uint16_t initial_time)
		: m_width(initial_width), m_height(initial_height), m_channel_count(initial_channel_count), m_time(initial_time)
	{
	}

	Shape::~Shape(void)
	{
	}
}
