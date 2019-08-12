#ifndef _GPL_PRIMITIVE_POINT_H_
#define _GPL_PRIMITIVE_POINT_H_

#include "gpl/primitives/base.h"
#include "gpl/gpl.h"

namespace Gpl
{
	namespace Primitive
	{
		class Point : public Base
		{
			private:
				Gra::Math::vec3<float> m_position;
				float m_size;
			public:
				static Gra::VertexBufferLayout s_vertex_buffer_layout;
				static Gra::IndexBuffer        s_index_buffer;

			public:
				Point(const Gra::Math::vec3<float>& initial_position, const Gra::Math::vec4<float>& initial_color, float initial_size);
				virtual ~Point(void);

			public:
				void draw(const Gra::Renderer& renderer, const glm::mat4& mvp) const override;

			public: // GETTERS
				const Gra::Math::vec3<float>& position(void) const;
				                      float   size     (void) const;
			public: // SETTERS
				void position(const Gra::Math::vec3<float>& new_position);
				void size    (                      float   new_size    );
		};

		// GETTERS
		inline const Gra::Math::vec3<float>& Point::position(void) const { return m_position; }
		inline                       float   Point::size    (void) const { return m_size;     }
		// SETTERS
		inline void Point::size(float new_size) { m_size = new_size; }
	}
}

#endif
