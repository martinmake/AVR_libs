#ifndef _GRA_RENDERER_H_
#define _GRA_RENDERER_H_

#include "gra/glstd.h"
#include "gra/gldebug.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "gra/graphics_objects/all.h"

namespace Gra
{
	class Renderer
	{
		private:
			GLFWwindow*  m_window;
			unsigned int m_width;
			unsigned int m_height;

		public:
			Renderer(void);
			Renderer(int width, int height, const std::string& title);
			~Renderer(void);

		public:
			void init(int width, int height, const std::string& title);

			template <typename T>
			void draw(const GraphicsObject::VertexArray& vertex_array, const GraphicsObject::Buffer::Index<T>& index_buffer, const GraphicsObject::Program& program, DrawMode mode) const;

			void start_frame() const;
			void   end_frame() const;

			bool should_close(void) const;

		public: // GETTERS
			unsigned int width (void) const;
			unsigned int height(void) const;
	};

	template <typename T>
	void Renderer::draw(const GraphicsObject::VertexArray& vertex_array, const GraphicsObject::Buffer::Index<T>& index_buffer, const GraphicsObject::Program& program, DrawMode mode) const
	{
		vertex_array.bind();
		index_buffer.bind();
		program     .bind();

		glCall(glDrawElements(DrawMode_to_GLenum(mode), index_buffer.indices().size(), GL_UNSIGNED_INT, nullptr));
	}

	inline bool Renderer::should_close(void) const { return glfwWindowShouldClose(m_window); }

	// GETTERS
	inline unsigned int Renderer::width (void) const { return m_width;  }
	inline unsigned int Renderer::height(void) const { return m_height; }
}

#endif
