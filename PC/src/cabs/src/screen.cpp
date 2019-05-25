#include <dialog.h>

#include "cabs/screen.h"

Screen::Screen(const Position& initial_position, const Size& initial_size)
	: m_position(initial_position), m_size(initial_size)
{
	m_win = newwin(m_size.h(), m_size.w(), m_position.y(), m_position.x());
}

Screen::Screen(void)
{
	m_win = stdscr;
}

Screen::~Screen(void)
{
	if (m_win != stdscr)
		delwin(m_win);
}

void Screen::redraw(void) const
{
	werase(m_win);

	draw();
}

void Screen::draw(void) const
{
	for (std::shared_ptr<Widget> widget : m_widgets)
		widget->draw();
}

void Screen::handle_key(int key)
{
	if (Cabs::move)
	{
		switch (key)
		{
			case KEY_LEFT:
			case 'H': break;

			case KEY_DOWN:
			case 'J': break;

			case KEY_UP:
			case 'K': break;

			case KEY_RIGHT:
			case 'L': break;

			case 'a':
			case 'A':
			case 'i':
			case 'I':
			case 'o':
			case 'O':
			case 'e': break;
		}
	}
}

Widget& Screen::operator[](int index)
{
	return *m_widgets[index];
}
