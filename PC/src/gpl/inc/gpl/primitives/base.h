#ifndef _GPL_PRIMITIVE_BASE_H_
#define _GPL_PRIMITIVE_BASE_H_

#include <vector>
#include <queue>

#include <gra/math.h>
#include <gra/renderer.h>
#include <gra/graphics_objects/program.h>
#include <gra/graphics_objects/vertex_array.h>
#include <gra/graphics_objects/buffers/vertex.h>

#include "gpl/core.h"
#include "gpl/events/primitive/all.h"
#include "gpl/primitive.h"
#include "gpl/data.h"

namespace Gpl
{
	namespace Primitive
	{
		class Base
		{
			protected: // CONSTRUCTORS
				Base(void);

			public: // GETTERS
				virtual bool colides(const Position& position) const;
				virtual bool is_container(void) const;
			public: // SETTERS
				void on_mouse_over(Event::Primitive::MouseOver::callback new_on_mouse_over);

			public: // FUNCTIONS
				virtual void draw(Data::Draw& data);
				// void on_mouse_over(std::queue<std::pair<Primitive::Base&, Data::MouseOver>>& queue);
				// virtual void on_mouse_over(Data::MouseOver& data);

			protected:
				Gra::GraphicsObject::VertexArray m_vertex_array;
				Gra::GraphicsObject::Program     m_program;
			protected:
				Event::Primitive::MouseOver::callback m_on_mouse_over;

			DECLARATION_MANDATORY_INTERFACE(Base)
		};

		// GETTERS
		inline bool Base::colides     (const Position& position) const { (void) position; assert(false); return false; }
		inline bool Base::is_container(void                    ) const {                                 return false; }
		// SETTERS
		DEFINITION_DEFAULT_SETTER(Base, on_mouse_over, Event::Primitive::MouseOver::callback)

		// FUNCTIONS
		inline void Base::draw         (Data::Draw     & data) { (void) data; }
		// inline void Base::on_mouse_over(Data::MouseOver& data) { (void) data; }

		DEFINITION_MANDATORY(Base, )
	}
}

#endif
