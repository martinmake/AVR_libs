#include <dialog.h>
#include <algorithm>

#include "cabs/screen.h"
#include "cabs/application.h"

Screen::Screen(void)
{
	m_win = derwin(stdscr, LINES - 1, COLS, 0, 0);
	m_status.attatch_to_window(m_win);
}

Screen::~Screen(void)
{
	delwin(m_win);
}

void Screen::draw(void) const
{
	wbkgd(m_win, application.screen_background_attr());
	for (std::shared_ptr<Widget> widget : m_widgets)
		widget->draw();
	m_status.draw();
}

void Screen::resize(void)
{
	wresize (m_win, LINES - 2, COLS);
	mvderwin(m_win, 0, 0);
	for (std::shared_ptr<Widget> widget : m_widgets)
		widget->resize();
	m_status.resize();
}

void Screen::handle_key(int key)
{
	if (Cabs::move)
	{
		switch (key)
		{
			using namespace Cabs;

			case KEY_LEFT:
			case 'h': move(Direction::LEFT);  break;

			case KEY_DOWN:
			case 'j': move(Direction::DOWN);  break;

			case KEY_UP:
			case 'k': move(Direction::UP);    break;

			case KEY_RIGHT:
			case 'l': move(Direction::RIGHT); break;

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

void Screen::move(Cabs::Direction direction)
{
	using namespace Cabs;

	std::vector<std::shared_ptr<Widget>> possible_widgets;
	std::vector<int>                     distance_metrics;

	int selected_x = m_selected_widget->position().x();
	int selected_y = m_selected_widget->position().y();
	switch (direction)
	{
		case Direction::LEFT:
			selected_y += m_selected_widget->size().h() / 2;
			break;
		case Direction::DOWN:
			selected_x += m_selected_widget->size().w() / 2;
			selected_y += m_selected_widget->size().h();
			break;
		case Direction::UP:
			selected_x += m_selected_widget->size().w() / 2;
			break;
		case Direction::RIGHT:
			selected_x += m_selected_widget->size().w();
			selected_y += m_selected_widget->size().h() / 2;
			break;
	}

	for (std::shared_ptr<Widget>& widget : m_widgets)
	{
		if (widget == m_selected_widget)
			continue;

		int other_x = widget->position().x();
		int other_y = widget->position().y();
		switch (direction)
		{
			case Direction::LEFT:
				other_x += widget->size().w();
				other_y += widget->size().h() / 2;
				break;
			case Direction::DOWN:
				other_x += widget->size().w() / 2;
				break;
			case Direction::UP:
				other_x += widget->size().w() / 2;
				other_y += widget->size().h();
				break;
			case Direction::RIGHT:
				other_y += widget->size().h() / 2;
				break;
		}

		bool is_in_wrong_direction = false;
		switch (direction)
		{
			case Direction::LEFT:  if (other_x >= selected_x) is_in_wrong_direction = true; break;
			case Direction::DOWN:  if (other_y <= selected_y) is_in_wrong_direction = true; break;
			case Direction::UP:    if (other_y >= selected_y) is_in_wrong_direction = true; break;
			case Direction::RIGHT: if (other_x <= selected_x) is_in_wrong_direction = true; break;
		}

		if (is_in_wrong_direction)
			continue;

		possible_widgets.emplace_back(widget);
		distance_metrics.emplace_back(abs(other_x - selected_x) + abs(other_y - selected_y));
	}

	if (possible_widgets.size() == 0)
		return;

	int closest_widget_index = min_element(distance_metrics.begin(), distance_metrics.end()) - distance_metrics.begin();
	m_selected_widget->is_selected(false);
	m_selected_widget = possible_widgets[closest_widget_index];
	m_selected_widget->is_selected(true);
	redraw();
}

Widget& Screen::operator[](int index)
{
	return *m_widgets[index];
}
