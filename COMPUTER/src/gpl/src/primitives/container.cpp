#include "gpl/primitives/container.h"
#include "gpl/primitive.h"

namespace Gpl
{
	namespace Primitive
	{
		using namespace Gra;
		using namespace Gra::GraphicsObject;
		using namespace Gra::Math;

		Container::Container(const Position& initial_position, const Size& initial_size)
			: m_position(initial_position), m_size(initial_size)
		{
		}

		Container::~Container(void)
		{
		}

		void Container::draw(std::queue<std::pair<Primitive::Container&, Data::Draw>>& queue)
		{
			queue.front().second.mvp = glm::translate(queue.front().second.mvp, glm::vec3(m_position.x, m_position.y, 0.0));
			for (std::unique_ptr<Primitive::Base>& primitive : m_primitives)
			{
				if (primitive->is_container())
					queue.push({ *((Primitive::Container*) &*primitive), queue.front().second });
				else
					primitive->draw(queue.front().second);
			}
		}

		bool Container::colides(const Position& position) const
		{
			if (position.x > m_position.x && position.x < m_position.x + m_size.x)
			if (position.y > m_position.y && position.y < m_position.y + m_size.y)
				return true;
			return false;
		}

		void Container::copy(const Container& other)
		{
			Primitive::Base::copy(other);

			m_position = other.m_position;
			m_size     = other.m_size;

			m_primitives.clear();
			for (const std::unique_ptr<Primitive::Base>& primitive : other.m_primitives)
				; // m_primitives.push_back(std::unique_ptr<Primitive::Base>(primitive->copy()));

		}
		void Container::move(Container&& other)
		{
			Primitive::Base::move(std::move(other));

			m_position   = std::move(other.m_position  );
			m_size       = std::move(other.m_size      );
			m_primitives = std::move(other.m_primitives);
		}
	}
}
