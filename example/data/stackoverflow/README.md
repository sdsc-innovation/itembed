
# Stack Overflow Tags

The dataset has been extracted using the [Stack Exchange Data Explorer](https://data.stackexchange.com/).
As described in the [documentation](https://data.stackexchange.com/help), the data is released under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/).
More information about licensing of content posted on Stack Overflow [here](https://stackoverflow.com/help/licensing).

The first one million questions with at least 4 tags were extracted:

```sql
SELECT Id, Tags
FROM Posts
WHERE LEN(Tags) - LEN(REPLACE(Tags, '<', '')) >= 4
ORDER BY Id
```
