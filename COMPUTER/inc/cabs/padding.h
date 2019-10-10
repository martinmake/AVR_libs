#ifndef _CABS_PADDING_H_
#define _CABS_PADDING_H_

#include <ncurses.h>

#include "cabs.h"

class Padding
{

	private:
		int m_top    = 0;
		int m_right  = 0;
		int m_bottom = 0;
		int m_left   = 0;

	public:
		Padding(int initial_top, int initial_right, int initial_bottom, int initial_left);
		Padding(void);
		~Padding(void);

	// GETTERS
	public:
		int top   (void) const;
		int right (void) const;
		int bottom(void) const;
		int left  (void) const;

	// SETTERS
	public:
		void top   (int new_top   );
		void right (int new_right );
		void bottom(int new_bottom);
		void left  (int new_left  );
};

// GETTERS
inline int Padding::top(void) const
{
	return m_top;
}
inline int Padding::right(void) const
{
	return m_right;
}
inline int Padding::bottom(void) const
{
	return m_bottom;
}
inline int Padding::left(void) const
{
	return m_left;
}

// SETTERS
inline void Padding::top(int new_top)
{
	m_top = new_top;
}
inline void Padding::right(int new_right)
{
	m_right = new_right;
}
inline void Padding::bottom(int new_bottom)
{
	m_bottom = new_bottom;
}
inline void Padding::left(int new_left)
{
	m_left = new_left;
}

#endif
