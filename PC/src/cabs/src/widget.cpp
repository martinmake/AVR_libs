#include "widget.h"

Widget::Widget(int x, int y, int w, int h)
	: m_x(x), m_y(y), m_w(w), m_h(h)
{
}

Widget::~Widget()
{
	delwin(m_win);
	delwin(m_win_shadow);
}

void Widget::attatch_to_window(WINDOW* win)
{
	using namespace Cabs::Position;

	int x, y;

	x = translate_x(win, m_x, m_w);
	y = translate_y(win, m_y, m_h);

	m_win        = derwin(win, m_h, m_w, y,     x);
	m_win_shadow = derwin(win, m_h, m_w, y + 1, x + 1);
}

void Widget::draw(void)
{
	if (m_shadow)
		wborder(m_win_shadow, ' ', 0, ' ', 0, ' ', 0, 0, 0);

	if (m_box)
		wborder(m_win, 0, 0, 0, 0, 0, 0, 0, 0);

	if (!m_label.empty())
		mvwprintw(m_win, 0, 2, "| %s |", m_label.c_str());

	wrefresh(m_win_shadow);
	wrefresh(m_win);
}
