#include <optional>

#include "primitives_shader.h"
#include "gpl/primitives/shapes/base.h"

namespace Gpl
{
	namespace Primitive
	{
		namespace Shape
		{
			using namespace Gra;
			using namespace GraphicsObject;

			Base::Base(const Gra::Math::vec4<float>& initial_color)
				: m_color(initial_color)
			{
				static std::unique_ptr<std::vector<Shader::Base>> s_shaders;
				if (!s_shaders)
				{
					s_shaders = std::make_unique<std::vector<Shader::Base>>();
					s_shaders->emplace_back(Shader::Vertex  (  VERTEX_SHADER));
					s_shaders->emplace_back(Shader::Fragment(FRAGMENT_SHADER));
				}

				static std::unique_ptr<Buffer::Vertex::Layout> s_vertex_buffer_layout;
				if (!s_vertex_buffer_layout)
				{
					s_vertex_buffer_layout = std::make_unique<Buffer::Vertex::Layout>();
					s_vertex_buffer_layout->push<float>(2);
				}

				m_program = Program(*s_shaders);
				m_vertex_array.layout(*s_vertex_buffer_layout);
			}

			Base::~Base(void)
			{
			}

			void Base::copy(const Base& other)
			{
				Primitive::Base::copy(other);

				m_color = other.m_color;
			}
			void Base::move(Base&& other)
			{
				Primitive::Base::move(std::move(other));

				m_color = std::move(other.m_color);
			}
		}
	}
}
