#include "gpl/primitives/base.h"

namespace Gpl
{
	namespace Primitive
	{
		Base::Base(const Gra::Math::vec4<float>& initial_color)
			: m_color(initial_color)
		{
		}
		Base::Base(Base&& other)
                        : m_color(std::move(other.m_color)), m_vertex_array(std::move(other.m_vertex_array)),  m_vertex_buffer(std::move(other.m_vertex_buffer))
		{
		}
		Base::~Base(void)
		{
		}
	}
}
