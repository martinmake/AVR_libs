#include <memory>
#include <utility>

#include <gra/shader.h>

#include "gpl/primitives/point.h"
#include "gpl/primitive.h"
#include "gpl/gpl.h"

namespace Gpl
{
	namespace Primitive
	{
		Gra::VertexBufferLayout Point::s_vertex_buffer_layout;
		Gra::IndexBuffer        Point::s_index_buffer;

		Point::Point(const Gra::Math::vec3<float>& initial_position, const Gra::Math::vec4<float>& initial_color, float initial_size)
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
		void Point::position(const Gra::Math::vec3<float>& new_position)
		{
			m_position = new_position;
			m_vertex_buffer.data(&m_position.x, 3 * sizeof(float));
		}
	}
}
