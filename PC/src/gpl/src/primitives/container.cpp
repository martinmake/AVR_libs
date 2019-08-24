#include "gpl/primitives/container.h"
#include "gpl/primitive.h"

namespace Gpl
{
	namespace Primitive
	{
		using namespace Gra;
		using namespace Gra::GraphicsObject;
		using namespace Gra::Math;

		Container::Container(const vec2<unsigned int>& initial_position, const vec2<unsigned int>& initial_size)
			: m_position(initial_position), m_size(initial_size)
		{
		}

		Container::~Container(void)
		{
		}

		void Container::draw(const vec2<unsigned int>& resolution, const glm::mat4& parent_mvp)
		{
			glm::mat4 mvp = glm::translate(parent_mvp, glm::vec3(m_position.x, m_position.y, 0.0));
			for (std::unique_ptr<Primitive::Base>& primitive : primitives)
				primitive->draw(m_size, mvp);
		}

		void Container::copy(const Container& other)
		{
			Primitive::Base::copy(other);

			m_position = other.m_position;
			m_size     = other.m_size;
		}
		void Container::move(Container&& other)
		{
			Primitive::Base::move(std::move(other));

			m_position = std::move(other.m_position);
			m_size     = std::move(other.m_size);
		}
	}
}
