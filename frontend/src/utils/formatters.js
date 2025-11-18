export const formatPHP = (value) => {
  return new Intl.NumberFormat('en-PH', {
    style: 'currency',
    currency: 'PHP',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
};

export const formatPeriodShort = (period) => {
  const [year, month] = period.split('-');
  const date = new Date(parseInt(year), parseInt(month) - 1);
  const monthShort = date.toLocaleDateString('en-US', { month: 'short' });
  const yearShort = year.slice(-2);
  return `${monthShort} '${yearShort}`;
};

