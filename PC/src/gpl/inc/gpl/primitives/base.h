#ifndef _GPL_PRIMITIVE_BASE_H_
#define _GPL_PRIMITIVE_BASE_H_

#include <vector>

#include <gra/math.h>
#include <gra/renderer.h>
#include <gra/graphics_objects/program.h>
#include <gra/graphics_objects/vertex_array.h>
#include <gra/graphics_objects/buffers/vertex.h>

#include "gpl/core.h"

namespace Gpl
{
	namespace Primitive
	{
		class Base
		{
			protected: // CONSTRUCTORS
				Base(void);

			public: // PUBLIC VARIABLES
				std::vector<std::unique_ptr<Primitive::Base>> primitives;

			public: // GETTERS
			public: // SETTERS

			public: // FUNCTIONS
				virtual void draw(const Gra::Math::vec2<unsigned int>& resolution, const glm::mat4& parent_mvp);

			public: // OPERATORS
				template <typename T> Primitive::Base& operator<<(const T&  primitive);
				template <typename T> Primitive::Base& operator<<(      T&& primitive);

			protected:
				Gra::GraphicsObject::VertexArray m_vertex_array;
				Gra::GraphicsObject::Program     m_program;

			DECLARATION_MANDATORY_INTERFACE(Base)
		};

		// GETTERS
		// SETTERS

		// FUNCTIONS
		inline void Base::draw(const Gra::Math::vec2<unsigned int>& resolution, const glm::mat4& parent_mvp) { (void) resolution; (void) parent_mvp; }

		// OPERATORS
		template <typename T> Primitive::Base& Primitive::Base::operator<<(const T&  primitive) { primitives.push_back(std::make_unique<T>(          primitive )); return *this; }
		template <typename T> Primitive::Base& Primitive::Base::operator<<(      T&& primitive) { primitives.push_back(std::make_unique<T>(std::move(primitive))); return *this; }

		DEFINITION_MANDATORY(Base, )
	}
}

#endif
