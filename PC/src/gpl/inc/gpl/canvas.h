#ifndef _GPL_CANVAS_H_
#define _GPL_CANVAS_H_

#include <gra/renderer.h>
#include <gra/window.h>

#include "gpl/core.h"
#include "gpl/primitives/container.h"

namespace Gpl
{
	class Canvas : public Gra::Window
	{
		public: // CONSTRUCTORS
			Canvas(int initial_width, int initial_height, const std::string initial_title);

		public: // GETTERS
			Primitive::Container& primitives(void);
		public: // SETTERS

		public: // FUNCTIONS
			void animate(void);

		public: // OPERATORS
			template <typename T> Canvas& operator<<(const T&  primitive);
			template <typename T> Canvas& operator<<(      T&& primitive);
			const Primitive::Base& operator[](uint16_t index) const;
			      Primitive::Base& operator[](uint16_t index);

		private:
			Primitive::Container m_primitives;

		DECLARATION_MANDATORY(Canvas)
	};

	// GETTERS
	inline Primitive::Container& Canvas::primitives(void) { return m_primitives; }

	// OPERATORS
	template <typename T> Canvas& Canvas::operator<<(const T&  primitive) { m_primitives <<           primitive;  return *this; }
	template <typename T> Canvas& Canvas::operator<<(      T&& primitive) { m_primitives << std::move(primitive); return *this; }
	inline const Primitive::Base& Canvas::operator[](uint16_t index) const { return m_primitives[index]; }
	inline       Primitive::Base& Canvas::operator[](uint16_t index)       { return m_primitives[index]; }

	DEFINITION_MANDATORY(Canvas, other.width(), other.height(), other.title())
}

#endif
