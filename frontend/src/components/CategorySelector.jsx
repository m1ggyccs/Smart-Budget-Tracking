import React from 'react';
import { categoryIcons } from '../utils/constants';

const CategorySelector = ({ categories, selectedCategory, onSelectCategory }) => {
  return (
    <div className="flex flex-wrap gap-3">
      {categories.map((cat) => (
        <button
          key={cat}
          onClick={() => onSelectCategory(cat)}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${
            selectedCategory === cat
              ? 'bg-gradient-to-r from-red-600 to-yellow-500 text-white shadow-lg shadow-red-500/30'
              : 'bg-slate-800 text-gray-400 hover:bg-slate-700 hover:text-white border border-blue-900'
          }`}
        >
          <span className="mr-2">{categoryIcons[cat] || '📦'}</span>
          {cat.charAt(0).toUpperCase() + cat.slice(1)}
        </button>
      ))}
    </div>
  );
};

export default CategorySelector;

