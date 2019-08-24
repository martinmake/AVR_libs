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

		void Base::copy(const Base& other)
		{
			m_vertex_array = other.m_vertex_array;
			m_program      = other.m_program;

			primitives.clear();
			for (const std::unique_ptr<Primitive::Base>& primitive : other.primitives)
				primitives.push_back(std::make_unique<Primitive::Base>(*primitive));
		}
		void Base::move(Base&& other)
		{
			m_vertex_array = std::move(other.m_vertex_array);
			m_program      = std::move(other.m_program     );
			primitives     = std::move(other.primitives    );
		}
	}
}
