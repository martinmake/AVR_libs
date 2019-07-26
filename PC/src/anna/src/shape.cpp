#include "shape.h"

namespace Anna
{

	Shape::Shape(void)
		: Shape(0, 0, 0)
	{
	}
	Shape::Shape(uint16_t initial_width, uint16_t initial_height, uint16_t initial_channel_count)
		: m_width(initial_width), m_height(initial_height), m_channel_count(initial_channel_count)
	{
	}

	Shape::~Shape(void)
	{
	}
}
