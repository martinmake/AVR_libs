#ifndef _GRA_RENDERER_H_
#define _GRA_RENDERER_H_

#include <functional>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "gra/glstd.h"
#include "gra/gldebug.h"
#include "gra/graphics_objects/all.h"
#include "gra/window.h"

namespace Gra
{
	class Renderer
	{
		public:
			 Renderer(void);
			~Renderer(void);

		public:
			void render(Window& window, std::function<void (void)> draw_calls) const;
			void draw(DrawMode mode, const GraphicsObject::Program& program, const GraphicsObject::VertexArray& vertex_array, const GraphicsObject::Buffer::Index & index_buffer) const;
			void draw(DrawMode mode, const GraphicsObject::Program& program, const GraphicsObject::VertexArray& vertex_array, uint32_t index_of_first_vertex = 0                ) const;
	};

	inline void Renderer::render(Window& window, std::function<void (void)> draw_calls) const
	{
		window.make_current();
		glViewport(0, 0, window.width(), window.height());

		window.clear();
		draw_calls();
		window.on_update();
	}

	inline void Renderer::draw(DrawMode mode, const GraphicsObject::Program& program, const GraphicsObject::VertexArray& vertex_array, const GraphicsObject::Buffer::Index& index_buffer) const
	{
		vertex_array.bind();
		index_buffer.bind();
		program     .bind();

		glCall(glDrawElements(DrawMode_to_GLenum(mode), index_buffer.indices().size(), GL_UNSIGNED_INT, nullptr));
	}
	inline void Renderer::draw(DrawMode mode, const GraphicsObject::Program& program, const GraphicsObject::VertexArray& vertex_array, uint32_t index_of_first_vertex) const
	{
		vertex_array.bind();
		program     .bind();

		glCall(glDrawArrays(DrawMode_to_GLenum(mode), index_of_first_vertex, vertex_array.count()));
	}
}

#endif
