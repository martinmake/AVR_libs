#include "gra/graphics_objects/shaders/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Shader
		{
			Base::Base(void)
			{
			}
			Base::Base(GLenum initial_type)
				: m_type(initial_type)
			{
				glCall(m_renderer_id = glCreateShader(m_type));
			}
			Base::Base(GLenum initial_type, const std::string& filepath_or_source)
				: Base(initial_type)
			{
				if (filepath_or_source.find('\n') == std::string::npos)
					load(filepath_or_source);
				else
					source(filepath_or_source);
			}

			Base::~Base(void)
			{
				if (m_renderer_id)
					glCall(glDeleteShader(m_renderer_id));
			}

			bool Base::load(const std::string& filepath)
			{
				std::ifstream stream(filepath);
				if (!stream.is_open())
				{
					std::cout << "[SHADER] SHADER NOT FOUND '" << filepath << '`' << std::endl;
					return false;
				}

				std::stringstream src;
				src << stream.rdbuf();

				return source(src.str());
			}


			static const char* type_to_string(GLenum type)
			{
				switch (type)
				{
					case GL_VERTEX_SHADER:   return "VERTEX";
					case GL_FRAGMENT_SHADER: return "FRAGMENT";
					case GL_GEOMETRY_SHADER: return "GEOMETRY";
					default:                 return "UNKNOWN";
				}
			}
			bool Base::source(const std::string& new_source)
			{
				{
					m_source = new_source;

					const char* source_ptr = m_source.c_str();
					glCall(glShaderSource(m_renderer_id, 1, &source_ptr, nullptr));
					glCall(glCompileShader(m_renderer_id));
				}
				{
					int result;
					glCall(glGetShaderiv(m_renderer_id, GL_COMPILE_STATUS, &result));
					if (result == GL_FALSE)
					{
						int length;
						glCall(glGetShaderiv(m_renderer_id, GL_INFO_LOG_LENGTH, &length));
						char* message = (char *) alloca(length * sizeof(char));
						glCall(glGetShaderInfoLog(m_renderer_id, length, &length, message));
						std::cout << "[SHADER] FAILED TO COMPILE '" << type_to_string(m_type) << "` SHADER" << std::endl;
						std::cout << message << std::endl;
						return false;
					}
				}

				return true;
			}

			void Base::copy(const Base& other)
			{
				GraphicsObject::Base::copy(other);

				source(other.m_source);
			}
			void Base::move(Base&& other)
			{
				GraphicsObject::Base::move(std::move(other));

				m_source = std::move(other.m_source);
			}
		}
	}
}
