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
			case 'h': break;

			case KEY_DOWN:
			case 'j': break;

			case KEY_UP:
			case 'k': break;

			case KEY_RIGHT:
			case 'l': break;

			case 'H': break;
			case 'J': break;
			case 'K': break;
			case 'L': break;
		}
	}
	else
	{
		if (m_selected_widget != nullptr)
			m_selected_widget->handle_key(key);
	}
}

Widget& Screen::operator[](int index)
{
	return *m_widgets[index];
}
