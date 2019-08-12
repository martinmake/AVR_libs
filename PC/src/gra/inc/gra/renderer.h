#ifndef _GRA_RENDERER_H_
#define _GRA_RENDERER_H_

#include "gra/glstd.h"
#include "gra/gldebug.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "gra/vertex_array.h"
#include "gra/vertex_buffer.h"
#include "gra/vertex_buffer_layout.h"
#include "gra/index_buffer.h"
#include "gra/shader.h"
#include "gra/texture.h"

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

			void draw(const VertexArray& vertex_array, const IndexBuffer& index_buffer, const Shader& shader, DrawMode mode) const;

			void start_frame() const;
			void   end_frame() const;

			bool should_close(void) const;

		public: // GETTERS
			unsigned int width (void) const;
			unsigned int height(void) const;
	};

	inline bool Renderer::should_close(void) const { return glfwWindowShouldClose(m_window); }

	// GETTERS
	inline unsigned int Renderer::width (void) const { return m_width;  }
	inline unsigned int Renderer::height(void) const { return m_height; }
}

#endif
