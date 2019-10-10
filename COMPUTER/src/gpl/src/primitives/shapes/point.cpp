#include <unistd.h>

#include "gpl/primitives/shapes/point.h"
#include "gpl/primitive.h"

namespace Gpl
{
	namespace Primitive
	{
		namespace Shape
		{
			using namespace Gra;
			using namespace GraphicsObject;

			static std::unique_ptr<Buffer::Vertex> g_vertex_buffer;

			Point::Point(const Position& initial_position, const Color& initial_color, unsigned int initial_size)
				: Base(initial_color), m_position(initial_position), m_size(initial_size)
			{
				if (!g_vertex_buffer)
				{
					float positions[] = { 0.0, 0.0 };
					g_vertex_buffer = std::make_unique<Buffer::Vertex>(positions, 2 * sizeof(float));
				}
				m_vertex_array.vertex_buffer(*g_vertex_buffer);
			}
			Point::~Point(void)
			{
			}

			bool Point::colides(const Position& position) const
			{
				unsigned int half_size = m_size / 2;
				if (std::abs((int) position.x - (int) m_position.x) < (int) half_size)
				if (std::abs((int) position.y - (int) m_position.y) < (int) half_size)
					return true;
				return false;
			}

			void Point::draw(Data::Draw& data)
			{
				static Renderer renderer;

				glm::mat4 mvp = glm::translate(data.mvp, glm::vec3(m_position.x, m_position.y, 0.0));

				m_program.set_uniform("u_mvp",                  mvp  );
				m_program.set_uniform("u_point_size", (float) m_size );
				m_program.set_uniform("u_color",              m_color);
				renderer.draw(DrawMode::POINTS, m_program, m_vertex_array);
			}

			void Point::copy(const Point& other)
			{
				Primitive::Base::copy(other);

				m_position = other.m_position;
				m_size     = other.m_size;
			}
			void Point::move(Point&& other)
			{
				Primitive::Base::move(std::move(other));

				m_position = other.m_position;
				m_size     = other.m_size;
			}
		}
	}
}
