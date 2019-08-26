#ifndef _GPL_CANVAS_H_
#define _GPL_CANVAS_H_

#include <gra/renderer.h>
#include <gra/window.h>

#include "gpl/core.h"
#include "gpl/primitives/container.h"

namespace Gpl
{
	class Canvas
	{
		public: // CONSTRUCTORS
			Canvas(int initial_width, int initial_height, const std::string initial_title);

		public: // GETTERS
			Primitive::Container& primitives(void);
			unsigned int width (void) const;
			unsigned int height(void) const;
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
			Gra::Window m_window;

		DECLARATION_MANDATORY(Canvas)
	};

	// GETTERS
	inline Primitive::Container& Canvas::primitives(void) { return m_primitives; }
	inline unsigned int Canvas::width (void) const { return m_window.width();  }
	inline unsigned int Canvas::height(void) const { return m_window.height(); }

	// OPERATORS
	template <typename T> Canvas& Canvas::operator<<(const T&  primitive) { m_primitives <<           primitive;  return *this; }
	template <typename T> Canvas& Canvas::operator<<(      T&& primitive) { m_primitives << std::move(primitive); return *this; }
	inline const Primitive::Base& Canvas::operator[](uint16_t index) const { return m_primitives[index]; }
	inline       Primitive::Base& Canvas::operator[](uint16_t index)       { return m_primitives[index]; }

	DEFINITION_MANDATORY(Canvas, other.m_window.width(), other.m_window.height(), other.m_window.title())
}

#endif
