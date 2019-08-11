#include <memory>

#include <gra/shader.h>

#include "primitives/point.h"
#include "primitive.h"
#include "gpl.h"

namespace Gpl
{
	namespace Primitive
	{
		Gra::VertexBufferLayout Point::s_vertex_buffer_layout;
		Gra::IndexBuffer        Point::s_index_buffer;

		Point::Point(const glm::vec3& initial_position, const glm::vec4& initial_color, float initial_size)
			: Base(initial_color), m_size(initial_size)
		{
			m_vertex_array.vertex_buffer(m_vertex_buffer       );
			m_vertex_array.layout       (s_vertex_buffer_layout);
			position(initial_position);
		}
		Point::~Point(void)
		{
		}

		void Point::draw(const Gra::Renderer& renderer, const glm::mat4& mvp) const
		{
			s_shader.set_uniform("u_mvp",          mvp);
			s_shader.set_uniform("u_point_size", m_size);
			s_shader.set_uniform("u_color",      m_color);
			renderer.draw(m_vertex_array, s_index_buffer, s_shader, Gra::DrawMode::POINTS);
		}

		// SETTERS
		void Point::position(const glm::vec3& new_position)
		{
			m_position = new_position;
			m_vertex_buffer.data(&m_position.x, 3 * sizeof(float));
		}
	}
}
