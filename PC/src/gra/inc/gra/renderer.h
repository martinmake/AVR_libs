#ifndef _GRA_RENDERER_H_
#define _GRA_RENDERER_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>

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
			GLFWwindow* m_window;
			unsigned int m_width,
				     m_height;

		public:
			Renderer(int width, int height, const std::string& title);
			~Renderer(void);

		public:
			void draw(const VertexArray& vertex_array, const IndexBuffer& index_buffer, const Shader& shader) const;
			void start_frame() const;
			void   end_frame() const;

			bool         should_close  (void) const;

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
