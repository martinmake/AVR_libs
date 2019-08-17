#include <sstream>

#include "anna/shape.h"

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

	// OPERATORS
	bool Shape::operator==(const Shape& other) const
	{
		if (m_width  == other.width ())
		if (m_height == other.height())
		if (m_depth  == other.depth ())
		if (m_time   == other.time  ())
			return true;

		return false;
	}
	Shape::operator std::string() const
	{
		std::stringstream output;

		output << m_width << ", " << m_height << ", " << m_depth << ", " << m_time;

		return output.str();
	}
}
