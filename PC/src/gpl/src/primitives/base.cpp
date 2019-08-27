#include "gpl/primitives/base.h"

namespace Gpl
{
	namespace Primitive
	{
		Base::Base(void)
		{
		}

		Base::~Base(void)
		{
		}

		void Base::on_mouse_over(Event::Primitive::MouseOver& event)
		{
			if (m_on_mouse_over)
				m_on_mouse_over(event, this);
		}

		void Base::copy(const Base& other)
		{
			m_vertex_array = other.m_vertex_array;
			m_program      = other.m_program;

			m_on_mouse_over = std::move(other.m_on_mouse_over);
		}
		void Base::move(Base&& other)
		{
			m_vertex_array = std::move(other.m_vertex_array);
			m_program      = std::move(other.m_program     );

			m_on_mouse_over = std::move(other.m_on_mouse_over);
		}
	}
}
